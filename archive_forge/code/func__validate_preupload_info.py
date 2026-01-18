import base64
import io
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
from tqdm.contrib.concurrent import thread_map
from huggingface_hub import get_session
from .constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
from .utils import tqdm as hf_tqdm
def _validate_preupload_info(preupload_info: dict):
    files = preupload_info.get('files')
    if not isinstance(files, list):
        raise ValueError('preupload_info is improperly formatted')
    for file_info in files:
        if not (isinstance(file_info, dict) and isinstance(file_info.get('path'), str) and isinstance(file_info.get('uploadMode'), str) and (file_info['uploadMode'] in ('lfs', 'regular'))):
            raise ValueError('preupload_info is improperly formatted:')
    return preupload_info