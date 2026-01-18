import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
def __makeAbsoluteUrl(self, url: str) -> str:
    if url.startswith('/'):
        url = f'{self.__prefix}{url}'
    else:
        o = urllib.parse.urlparse(url)
        assert o.hostname in [self.__hostname, 'uploads.github.com', 'status.github.com', 'github.com'], o.hostname
        assert o.path.startswith((self.__prefix, self.__graphql_prefix, '/api/')), o.path
        assert o.port == self.__port, o.port
        url = o.path
        if o.query != '':
            url += f'?{o.query}'
    return url