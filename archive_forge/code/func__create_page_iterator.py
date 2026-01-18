import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def _create_page_iterator(conf: Config, url: str, method: str, params: Mapping[str, str]) -> Iterator[Dict[str, Any]]:
    p = dict(params).copy()
    while True:
        req = Request(url=url, method=method, params=p, success_codes=(200, 404))
        resp = execute_api_request(conf, req)
        if resp.status == 404:
            return
        result = json.loads(resp.data)
        yield result
        if 'nextPageToken' not in result:
            break
        p['pageToken'] = result['nextPageToken']