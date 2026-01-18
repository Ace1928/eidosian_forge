from __future__ import annotations
import itertools
import types
import warnings
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
class ResultException(Exception):
    pass