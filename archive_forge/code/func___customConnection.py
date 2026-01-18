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
def __customConnection(self, url: str) -> Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]:
    cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]] = None
    if not url.startswith('/'):
        o = urllib.parse.urlparse(url)
        if o.hostname != self.__hostname or (o.port and o.port != self.__port) or (o.scheme != self.__scheme and (not (o.scheme == 'https' and self.__scheme == 'http'))):
            if o.scheme == 'http':
                cnx = self.__httpConnectionClass(o.hostname, o.port, retry=self.__retry, pool_size=self.__pool_size)
                self.__custom_connections.append(cnx)
            elif o.scheme == 'https':
                cnx = self.__httpsConnectionClass(o.hostname, o.port, retry=self.__retry, pool_size=self.__pool_size)
                self.__custom_connections.append(cnx)
    return cnx