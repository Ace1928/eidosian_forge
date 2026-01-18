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
class GunzipFilter:
    """WSGI middleware that unzips gzip-encoded requests before
    passing on to the underlying application.
    """

    def __init__(self, application) -> None:
        self.app = application

    def __call__(self, environ, start_response):
        import gzip
        if environ.get('HTTP_CONTENT_ENCODING', '') == 'gzip':
            environ['wsgi.input'] = gzip.GzipFile(filename=None, fileobj=environ['wsgi.input'], mode='rb')
            del environ['HTTP_CONTENT_ENCODING']
            if 'CONTENT_LENGTH' in environ:
                del environ['CONTENT_LENGTH']
        return self.app(environ, start_response)