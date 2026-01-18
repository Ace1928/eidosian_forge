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
class StreamingReadFile(BaseStreamingReadFile):

    def __init__(self, conf: Config, path: str, size: Optional[int]) -> None:
        if size is None:
            st = maybe_stat(conf, path)
            if st is None:
                raise FileNotFoundError(f"No such file or bucket: '{path}'")
            size = st.size
        super().__init__(conf=conf, path=path, size=size)

    def _request_chunk(self, streaming: bool, start: int, end: Optional[int]=None) -> 'urllib3.BaseHTTPResponse':
        bucket, name = split_path(self._path)
        req = Request(url=build_url('/storage/v1/b/{bucket}/o/{name}', bucket=bucket, name=name), method='GET', params=dict(alt='media'), headers={'Range': common.calc_range(start=start, end=end)}, success_codes=(206, 416), preload_content=not streaming)
        return execute_api_request(self._conf, req)