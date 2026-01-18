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
def _extract_name(self, url, service_type=None, project_id=None):
    url_path = parse.urlparse(url).path.strip()
    if url_path.startswith('/'):
        url_path = url_path[1:]
    url_parts = [x for x in url_path.split('/') if x != project_id and (not project_id or (project_id and x != 'AUTH_' + project_id))]
    if url_parts[0] and url_parts[0][0] == 'v' and url_parts[0][1] and url_parts[0][1].isdigit():
        url_parts = url_parts[1:]
    parts = [part for part in url_parts if part]
    if not parts:
        return ['account']
    if len(parts) == 1:
        if 'endpoints' in parts:
            return ['endpoints']
        else:
            return ['container']
    else:
        return ['object']