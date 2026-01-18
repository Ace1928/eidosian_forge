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
def _url_to_path(url):
    return url.replace('/', os.path.sep)