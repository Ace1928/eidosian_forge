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