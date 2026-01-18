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
def _can_access_container(conf: Config, account: str, container: str, auth: Tuple[str, str], out_failures: List[RequestFailure]) -> bool:
    success_codes = [200, 403, 404, INVALID_HOSTNAME_STATUS]
    if auth[0] == ANONYMOUS:
        success_codes.append(409)

    def build_req() -> Request:
        req = Request(method='GET', url=build_url(account, '/{container}', container=container), params={'restype': 'container', 'comp': 'list', 'maxresults': '1'}, success_codes=success_codes)
        return create_api_request(req, auth=auth)
    resp = common.execute_request(conf, build_req)
    if resp.status == INVALID_HOSTNAME_STATUS:
        return True
    if resp.status == 404 and auth[0] == ANONYMOUS:
        out_failures.append(RequestFailure.create_from_request_response('Could not access container', build_req(), resp))
        return False
    if resp.status in (200, 404):
        return True
    else:
        out_failures.append(RequestFailure.create_from_request_response('Could not access container', build_req(), resp))
        return False