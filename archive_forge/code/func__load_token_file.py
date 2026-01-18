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
def _load_token_file(self):
    if os.path.exists(self._access_token_file):
        with open(self._access_token_file) as f:
            tokens = json.load(f, object_hook=hinted_tuple_hook)
            for key, value in zip(tokens['token_keys'], tokens['token_values']):
                self._tokens[key] = value
            for key, value in zip(tokens['expiration_keys'], tokens['expiration_values']):
                self._expirations[key] = value