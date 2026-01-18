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
class StringPayload(BytesPayload):

    def __init__(self, value: str, *args: Any, encoding: Optional[str]=None, content_type: Optional[str]=None, **kwargs: Any) -> None:
        if encoding is None:
            if content_type is None:
                real_encoding = 'utf-8'
                content_type = 'text/plain; charset=utf-8'
            else:
                mimetype = parse_mimetype(content_type)
                real_encoding = mimetype.parameters.get('charset', 'utf-8')
        else:
            if content_type is None:
                content_type = 'text/plain; charset=%s' % encoding
            real_encoding = encoding
        super().__init__(value.encode(real_encoding), *args, encoding=real_encoding, content_type=content_type, **kwargs)