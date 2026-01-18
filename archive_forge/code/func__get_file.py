from datetime import datetime
import json
import logging
import numpy as np
import os
from urllib.parse import urlparse
import time
from ray.air._internal.json import SafeFallbackEncoder
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.output_writer import OutputWriter
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.compression import pack, compression_supported
from ray.rllib.utils.typing import FileType, SampleBatchType
from typing import Any, Dict, List
def _get_file(self) -> FileType:
    if not self.cur_file or self.bytes_written >= self.max_file_size:
        if self.cur_file:
            self.cur_file.close()
        timestr = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(self.path, 'output-{}_worker-{}_{}.json'.format(timestr, self.ioctx.worker_index, self.file_index))
        if self.path_is_uri:
            if smart_open is None:
                raise ValueError('You must install the `smart_open` module to write to URIs like {}'.format(path))
            self.cur_file = smart_open(path, 'w')
        else:
            self.cur_file = open(path, 'w')
        self.file_index += 1
        self.bytes_written = 0
        logger.info('Writing to new output file {}'.format(self.cur_file))
    return self.cur_file