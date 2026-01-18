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
def _bulk_delete(self, elements):
    data = '\n'.join([parse.quote(x) for x in elements])
    self.delete('?bulk-delete', data=data, headers={'Content-Type': 'text/plain', 'Accept': 'application/json'})