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
def __structuredFromJson(self, data: str) -> Any:
    if len(data) == 0:
        return None
    else:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        try:
            return json.loads(data)
        except ValueError:
            if data.startswith('{') or data.startswith('['):
                raise
            return {'data': data}