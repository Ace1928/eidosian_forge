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
def _extract_error(data: bytes) -> Tuple[Optional[str], Optional[str]]:
    if data.startswith(b'\xef\xbb\xbf<?xml'):
        try:
            result = xml.parse(data)
            return (result['Error']['Code'], result['Error'].get('Message'))
        except Exception:
            pass
    elif data.startswith(b'{'):
        try:
            result = json.loads(data)
            return (str(result['error']), result.get('error_description'))
        except Exception:
            pass
    return (None, None)