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
class Payload(ABC):
    _default_content_type: str = 'application/octet-stream'
    _size: Optional[int] = None

    def __init__(self, value: Any, headers: Optional[Union[_CIMultiDict, Dict[str, str], Iterable[Tuple[str, str]]]]=None, content_type: Union[str, None, _SENTINEL]=sentinel, filename: Optional[str]=None, encoding: Optional[str]=None, **kwargs: Any) -> None:
        self._encoding = encoding
        self._filename = filename
        self._headers: _CIMultiDict = CIMultiDict()
        self._value = value
        if content_type is not sentinel and content_type is not None:
            self._headers[hdrs.CONTENT_TYPE] = content_type
        elif self._filename is not None:
            content_type = mimetypes.guess_type(self._filename)[0]
            if content_type is None:
                content_type = self._default_content_type
            self._headers[hdrs.CONTENT_TYPE] = content_type
        else:
            self._headers[hdrs.CONTENT_TYPE] = self._default_content_type
        self._headers.update(headers or {})

    @property
    def size(self) -> Optional[int]:
        """Size of the payload."""
        return self._size

    @property
    def filename(self) -> Optional[str]:
        """Filename of the payload."""
        return self._filename

    @property
    def headers(self) -> _CIMultiDict:
        """Custom item headers"""
        return self._headers

    @property
    def _binary_headers(self) -> bytes:
        return ''.join([k + ': ' + v + '\r\n' for k, v in self.headers.items()]).encode('utf-8') + b'\r\n'

    @property
    def encoding(self) -> Optional[str]:
        """Payload encoding"""
        return self._encoding

    @property
    def content_type(self) -> str:
        """Content type"""
        return self._headers[hdrs.CONTENT_TYPE]

    def set_content_disposition(self, disptype: str, quote_fields: bool=True, _charset: str='utf-8', **params: Any) -> None:
        """Sets ``Content-Disposition`` header."""
        self._headers[hdrs.CONTENT_DISPOSITION] = content_disposition_header(disptype, quote_fields=quote_fields, _charset=_charset, **params)

    @abstractmethod
    async def write(self, writer: AbstractStreamWriter) -> None:
        """Write payload.

        writer is an AbstractStreamWriter instance:
        """