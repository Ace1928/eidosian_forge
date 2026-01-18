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
def _save_token_file(self, log_callback: Callable[[str], None]):
    os.makedirs(os.path.dirname(self._access_lock_file), exist_ok=True)
    try:
        with filelock.FileLock(self._access_lock_file, timeout=1):
            os.makedirs(os.path.dirname(self._access_token_file), exist_ok=True)
            tmp_path = self._access_token_file + '.tmp'
            with open(tmp_path, 'w') as f:
                f.write(TupleEncoder().encode({'token_keys': list(self._tokens.keys()), 'token_values': list(self._tokens.values()), 'expiration_keys': list(self._expirations.keys()), 'expiration_values': list(self._expirations.values())}))
            os.replace(tmp_path, self._access_token_file)
    except filelock.Timeout:
        log_callback('Another instance of this application currently holds the lock.')