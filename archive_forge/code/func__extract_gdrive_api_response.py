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
def _extract_gdrive_api_response(response, chunk_size: int=32 * 1024) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)
    first_chunk = None
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)
    try:
        match = re.search('<title>Google Drive - (?P<api_response>.+?)</title>', first_chunk.decode())
        api_response = match['api_response'] if match is not None else None
    except UnicodeDecodeError:
        api_response = None
    return (api_response, content)