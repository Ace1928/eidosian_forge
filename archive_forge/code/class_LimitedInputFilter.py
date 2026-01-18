import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class LimitedInputFilter:
    """WSGI middleware that limits the input length of a request to that
    specified in Content-Length.
    """

    def __init__(self, application) -> None:
        self.app = application

    def __call__(self, environ, start_response):
        content_length = environ.get('CONTENT_LENGTH', '')
        if content_length:
            environ['wsgi.input'] = _LengthLimitedFile(environ['wsgi.input'], int(content_length))
        return self.app(environ, start_response)