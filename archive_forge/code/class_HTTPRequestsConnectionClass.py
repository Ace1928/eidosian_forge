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
class HTTPRequestsConnectionClass:

    def __init__(self, host: str, port: Optional[int]=None, strict: bool=False, timeout: Optional[int]=None, retry: Optional[Union[int, Retry]]=None, pool_size: Optional[int]=None, **kwargs: Any):
        self.port = port if port else 80
        self.host = host
        self.protocol = 'http'
        self.timeout = timeout
        self.verify = kwargs.get('verify', True)
        self.session = requests.Session()
        self.session.auth = Requester.noopAuth
        if retry is None:
            self.retry = requests.adapters.DEFAULT_RETRIES
        else:
            self.retry = retry
        if pool_size is None:
            self.pool_size = requests.adapters.DEFAULT_POOLSIZE
        else:
            self.pool_size = pool_size
        self.adapter = requests.adapters.HTTPAdapter(max_retries=self.retry, pool_connections=self.pool_size, pool_maxsize=self.pool_size)
        self.session.mount('http://', self.adapter)

    def request(self, verb: str, url: str, input: None, headers: Dict[str, str]) -> None:
        self.verb = verb
        self.url = url
        self.input = input
        self.headers = headers

    def getresponse(self) -> RequestsResponse:
        verb = getattr(self.session, self.verb.lower())
        url = f'{self.protocol}://{self.host}:{self.port}{self.url}'
        r = verb(url, headers=self.headers, data=self.input, timeout=self.timeout, verify=self.verify, allow_redirects=False)
        return RequestsResponse(r)

    def close(self) -> None:
        self.session.close()