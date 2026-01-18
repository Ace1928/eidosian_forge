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
class StringIOPayload(StringPayload):

    def __init__(self, value: IO[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(value.read(), *args, **kwargs)