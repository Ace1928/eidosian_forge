import logging
import math
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Optional
import zipfile
import ray.data
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import from_json_data, postprocess_actions
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
def _unzip_if_needed(paths: List[str], format: str):
    """If a path in paths is a zip file, unzip it and use path of the unzipped file"""
    ret_paths = []
    for path in paths:
        if re.search('\\.zip$', str(path)):
            if str(path).startswith('s3://'):
                raise ValueError('unzip_if_needed currently does not support remote paths from s3')
            extract_path = './'
            try:
                _unzip_this_path(str(path), extract_path)
            except FileNotFoundError:
                try:
                    _unzip_this_path(Path(__file__).parent.parent / path, extract_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f'File not found: {path}')
            unzipped_path = str(Path(extract_path).absolute() / f'{Path(path).stem}.{format}')
            ret_paths.append(unzipped_path)
        elif str(path).startswith('s3://'):
            ret_paths.append(path)
        else:
            if not Path(path).exists():
                relative_path = str(Path(__file__).parent.parent / path)
                if not Path(relative_path).exists():
                    raise FileNotFoundError(f'File not found: {path}')
                path = relative_path
            ret_paths.append(path)
    return ret_paths