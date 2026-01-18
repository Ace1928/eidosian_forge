import bz2
import contextlib
import gzip
import hashlib
import itertools
import lzma
import os
import os.path
import pathlib
import re
import sys
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import Any, Callable, Dict, IO, Iterable, Iterator, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse
import numpy as np
import requests
import torch
from torch.utils.model_zoo import tqdm
from .._internally_replaced_utils import _download_file_from_remote_location, _is_remote_location_available
def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)
    if re.match('(drive|docs)[.]google[.]com', parts.netloc) is None:
        return None
    match = re.match('/file/d/(?P<id>[^/]*)', parts.path)
    if match is None:
        return None
    return match.group('id')