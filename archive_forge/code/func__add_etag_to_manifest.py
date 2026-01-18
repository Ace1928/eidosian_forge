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
def _add_etag_to_manifest(self, segment_results, manifest):
    for result in segment_results:
        if 'Etag' not in result.headers:
            continue
        name = self._object_name_from_url(result.url)
        for entry in manifest:
            if entry['path'] == '/{name}'.format(name=parse.unquote(name)):
                entry['etag'] = result.headers['Etag']