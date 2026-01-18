import glob
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
import zipfile
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType
@PublicAPI
class JsonReader(InputReader):
    """Reader object that loads experiences from JSON file chunks.

    The input files will be read from in random order.
    """

    @PublicAPI
    def __init__(self, inputs: Union[str, List[str]], ioctx: Optional[IOContext]=None):
        """Initializes a JsonReader instance.

        Args:
            inputs: Either a glob expression for files, e.g. `/tmp/**/*.json`,
                or a list of single file paths or URIs, e.g.,
                ["s3://bucket/file.json", "s3://bucket/file2.json"].
            ioctx: Current IO context object or None.
        """
        logger.info('You are using JSONReader. It is recommended to use ' + 'DatasetReader instead for better sharding support.')
        self.ioctx = ioctx or IOContext()
        self.default_policy = self.policy_map = None
        self.batch_size = 1
        if self.ioctx:
            self.batch_size = self.ioctx.config.get('train_batch_size', 1)
            num_workers = self.ioctx.config.get('num_workers', 0)
            if num_workers:
                self.batch_size = max(math.ceil(self.batch_size / num_workers), 1)
        if self.ioctx.worker is not None:
            self.policy_map = self.ioctx.worker.policy_map
            self.default_policy = self.policy_map.get(DEFAULT_POLICY_ID)
        if isinstance(inputs, str):
            inputs = os.path.abspath(os.path.expanduser(inputs))
            if os.path.isdir(inputs):
                inputs = [os.path.join(inputs, '*.json'), os.path.join(inputs, '*.zip')]
                logger.warning(f'Treating input directory as glob patterns: {inputs}')
            else:
                inputs = [inputs]
            if any((urlparse(i).scheme not in [''] + WINDOWS_DRIVES for i in inputs)):
                raise ValueError("Don't know how to glob over `{}`, ".format(inputs) + 'please specify a list of files to read instead.')
            else:
                self.files = []
                for i in inputs:
                    self.files.extend(glob.glob(i))
        elif isinstance(inputs, (list, tuple)):
            self.files = list(inputs)
        else:
            raise ValueError('type of inputs must be list or str, not {}'.format(inputs))
        if self.files:
            logger.info('Found {} input files.'.format(len(self.files)))
        else:
            raise ValueError('No files found matching {}'.format(inputs))
        self.cur_file = None

    @override(InputReader)
    def next(self) -> SampleBatchType:
        ret = []
        count = 0
        while count < self.batch_size:
            batch = self._try_parse(self._next_line())
            tries = 0
            while not batch and tries < 100:
                tries += 1
                logger.debug('Skipping empty line in {}'.format(self.cur_file))
                batch = self._try_parse(self._next_line())
            if not batch:
                raise ValueError('Failed to read valid experience batch from file: {}'.format(self.cur_file))
            batch = self._postprocess_if_needed(batch)
            count += batch.count
            ret.append(batch)
        ret = concat_samples(ret)
        return ret

    def read_all_files(self) -> SampleBatchType:
        """Reads through all files and yields one SampleBatchType per line.

        When reaching the end of the last file, will start from the beginning
        again.

        Yields:
            One SampleBatch or MultiAgentBatch per line in all input files.
        """
        for path in self.files:
            file = self._try_open_file(path)
            while True:
                line = file.readline()
                if not line:
                    break
                batch = self._try_parse(line)
                if batch is None:
                    break
                yield batch

    def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
        if not self.ioctx.config.get('postprocess_inputs'):
            return batch
        batch = convert_ma_batch_to_sample_batch(batch)
        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                out.append(self.default_policy.postprocess_trajectory(sub_batch))
            return concat_samples(out)
        else:
            raise NotImplementedError('Postprocessing of multi-agent data not implemented yet.')

    def _try_open_file(self, path):
        if urlparse(path).scheme not in [''] + WINDOWS_DRIVES:
            if smart_open is None:
                raise ValueError('You must install the `smart_open` module to read from URIs like {}'.format(path))
            ctx = smart_open
        else:
            if path.startswith('~/'):
                path = os.path.join(os.environ.get('HOME', ''), path[2:])
            path_orig = path
            if not os.path.exists(path):
                path = os.path.join(Path(__file__).parent.parent, path)
            if not os.path.exists(path):
                raise FileNotFoundError(f'Offline file {path_orig} not found!')
            if re.search('\\.zip$', path):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(Path(path).parent)
                path = re.sub('\\.zip$', '.json', path)
                assert os.path.exists(path)
            ctx = open
        file = ctx(path, 'r')
        return file

    def _try_parse(self, line: str) -> Optional[SampleBatchType]:
        line = line.strip()
        if not line:
            return None
        try:
            batch = self._from_json(line)
        except Exception:
            logger.exception('Ignoring corrupt json record in {}: {}'.format(self.cur_file, line))
            return None
        batch = postprocess_actions(batch, self.ioctx)
        return batch

    def _next_line(self) -> str:
        if not self.cur_file:
            self.cur_file = self._next_file()
        line = self.cur_file.readline()
        tries = 0
        while not line and tries < 100:
            tries += 1
            if hasattr(self.cur_file, 'close'):
                self.cur_file.close()
            self.cur_file = self._next_file()
            line = self.cur_file.readline()
            if not line:
                logger.debug('Ignoring empty file {}'.format(self.cur_file))
        if not line:
            raise ValueError('Failed to read next line from files: {}'.format(self.files))
        return line

    def _next_file(self) -> FileType:
        if self.cur_file is None and self.ioctx.worker is not None:
            idx = self.ioctx.worker.worker_index
            total = self.ioctx.worker.num_workers or 1
            path = self.files[round((len(self.files) - 1) * (idx / total))]
        else:
            path = random.choice(self.files)
        return self._try_open_file(path)

    def _from_json(self, data: str) -> SampleBatchType:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        json_data = json.loads(data)
        return from_json_data(json_data, self.ioctx.worker)