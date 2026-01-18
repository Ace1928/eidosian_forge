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
def __deferRequest(self, verb: str) -> None:
    requests = self.__last_requests.values()
    writes = [l for v, l in self.__last_requests.items() if v != 'GET']
    last_request = max(requests) if requests else 0
    last_write = max(writes) if writes else 0
    next_request = last_request + self.__seconds_between_requests if self.__seconds_between_requests else 0
    next_write = last_write + self.__seconds_between_writes if self.__seconds_between_writes else 0
    next = next_request if verb == 'GET' else max(next_request, next_write)
    defer = max(next - datetime.now(timezone.utc).timestamp(), 0)
    if defer > 0:
        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)
        self.__logger.debug(f'sleeping {defer}s before next GitHub request')
        time.sleep(defer)