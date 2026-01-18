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
@runtime_checkable
class BlobPathLike(Protocol):
    """Similar to the __fspath__ protocol, but for remote blob paths."""

    def __blobpath__(self) -> str:
        ...