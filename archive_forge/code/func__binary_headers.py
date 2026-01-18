import asyncio
import enum
import io
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
@property
def _binary_headers(self) -> bytes:
    return ''.join([k + ': ' + v + '\r\n' for k, v in self.headers.items()]).encode('utf-8') + b'\r\n'