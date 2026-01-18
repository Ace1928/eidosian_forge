import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def _finalize_blob(conf: Config, path: str, url: str, block_ids: List[str], md5_digest: Optional[bytes], version: Optional[str]) -> None:
    body = {'BlockList': {'Latest': block_ids}}
    headers = {}
    if md5_digest is not None:
        headers['x-ms-blob-content-md5'] = base64.b64encode(md5_digest).decode('utf8')
    if version is not None:
        headers['If-Match'] = version
    req = Request(url=url, method='PUT', headers=headers, params=dict(comp='blocklist'), data=body, success_codes=(201, 400, 404, 412, INVALID_HOSTNAME_STATUS))
    resp = execute_api_request(conf, req)
    if resp.status == 400:
        result = xml.parse(resp.data)
        if result['Error']['Code'] == 'InvalidBlockList':
            raise ConcurrentWriteFailure.create_from_request_response(f'Invalid block list, most likely a concurrent writer wrote to the same path: `{path}`', request=req, response=resp)
        else:
            raise RequestFailure.create_from_request_response(message=f'unexpected status {resp.status}', request=req, response=resp)
    elif resp.status == 404 or resp.status == INVALID_HOSTNAME_STATUS:
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    elif resp.status == 412:
        if resp.headers['x-ms-error-code'] != 'ConditionNotMet':
            raise RequestFailure.create_from_request_response(message=f'unexpected status {resp.status}', request=req, response=resp)
        else:
            raise VersionMismatch.create_from_request_response(message='etag mismatch', request=req, response=resp)