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
def _get_storage_account_id(conf: Config, subscription_id: str, account: str, auth: Tuple[str, str]) -> Optional[str]:
    url = f'https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Storage/storageAccounts'
    params = {'api-version': '2019-04-01'}
    while True:

        def build_req() -> Request:
            req = Request(method='GET', url=url, params=params, success_codes=(200, 401, 403))
            return create_api_request(req, auth=auth)
        resp = common.execute_request(conf, build_req)
        if resp.status in (401, 403):
            return None
        out = json.loads(resp.data)
        for obj in out['value']:
            if obj['name'] == account:
                return obj['id']
        if 'nextLink' not in out:
            return None
        url = out['nextLink']
        params = None