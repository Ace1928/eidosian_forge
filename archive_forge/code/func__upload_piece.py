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
def _upload_piece(self, offset: int, piece: memoryview, finalize: bool) -> None:
    start = offset
    end = offset + len(piece) - 1
    total_size = '*'
    if finalize:
        total_size = offset + len(piece)
    headers = {'Content-Type': 'application/octet-stream', 'Content-Range': f'bytes {start}-{end}/{total_size}'}
    if len(piece) == 0 and finalize:
        headers['Content-Range'] = f'bytes */{total_size}'
    req = Request(url=self._upload_url, data=piece, headers=headers, method='PUT', success_codes=(200, 201) if finalize else (308,))
    try:
        execute_api_request(self._conf, req)
    except RequestFailure as e:
        if e.response_status in (404, 410):
            raise RestartableStreamingWriteFailure(message=e.message, request_string=e.request_string, response_status=e.response_status, error=e.error, error_description=e.error_description)
        else:
            raise