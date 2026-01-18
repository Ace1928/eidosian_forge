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
def _flip_byte_order(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).view(*t.shape, t.element_size()).flip(-1).view(*t.shape[:-1], -1).view(t.dtype)