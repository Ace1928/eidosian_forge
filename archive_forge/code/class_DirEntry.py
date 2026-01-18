import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
class DirEntry(NamedTuple):
    path: str
    name: str
    is_dir: bool
    is_file: bool
    stat: Optional[Stat]