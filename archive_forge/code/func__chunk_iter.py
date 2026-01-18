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
def _chunk_iter(f):
    while True:
        line = f.readline()
        length = int(line.rstrip(), 16)
        chunk = f.read(length + 2)
        if length == 0:
            break
        yield chunk[:-2]