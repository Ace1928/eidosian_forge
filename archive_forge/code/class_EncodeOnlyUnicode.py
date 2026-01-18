from __future__ import annotations
import itertools
import types
import warnings
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
class EncodeOnlyUnicode:
    """
    This is a crappy work-around for
    http://bugs.python.org/issue11649


    """

    def __init__(self, stream: BinaryIO):
        self.__stream = stream

    def write(self, arg):
        if isinstance(arg, str):
            self.__stream.write(arg.encode('utf-8'))
        else:
            self.__stream.write(arg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__stream, name)