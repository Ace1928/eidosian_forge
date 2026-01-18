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
def _get_storage_account_key(conf: Config, account: str, container: str, creds: Mapping[str, Any], out_failures: List[RequestFailure]) -> Optional[Tuple[Any, float]]:

    def build_req_access_token() -> Request:
        return _create_access_token_request(creds=creds, scope='https://management.azure.com/.default')
    resp = common.execute_request(conf, build_req_access_token)
    result = json.loads(resp.data)
    auth = (OAUTH_TOKEN, result['access_token'])
    stored_subscription_ids = load_subscription_ids()
    storage_account_id = None
    for subscription_id in stored_subscription_ids:
        storage_account_id = _get_storage_account_id(conf, subscription_id, account, auth)
        if storage_account_id is not None:
            break
    else:
        subscription_ids = _get_subscription_ids(conf, auth)
        unchecked_subscription_ids = [id for id in subscription_ids if id not in stored_subscription_ids]
        for subscription_id in unchecked_subscription_ids:
            storage_account_id = _get_storage_account_id(conf, subscription_id, account, auth)
            if storage_account_id is not None:
                break
        else:
            return None

    def build_req_list_keys() -> Request:
        req = Request(method='POST', url=f'https://management.azure.com{storage_account_id}/listKeys', params={'api-version': '2019-04-01'})
        return create_api_request(req, auth=auth)
    resp = common.execute_request(conf, build_req_list_keys)
    result = json.loads(resp.data)
    for key in result['keys']:
        if key['permissions'] == 'FULL':
            storage_key_auth = (SHARED_KEY, key['value'])
            if _can_access_container(conf, account, container, storage_key_auth, out_failures=out_failures):
                return storage_key_auth
            else:
                raise Error(f"Found storage account key, but it was unable to access storage account: '{account}' and container: '{container}'")
    raise Error(f"Storage account was found, but storage account keys were missing: '{account}'")