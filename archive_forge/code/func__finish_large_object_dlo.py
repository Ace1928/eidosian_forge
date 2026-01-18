from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def _finish_large_object_dlo(self, endpoint, headers):
    headers = headers.copy()
    headers['X-Object-Manifest'] = endpoint
    retries = 3
    while True:
        try:
            return exceptions.raise_from_response(self.put(endpoint, headers=headers))
        except Exception:
            retries -= 1
            if retries == 0:
                raise