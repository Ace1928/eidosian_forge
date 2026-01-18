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
def _get_container_name(self, obj=None, container=None):
    if obj is not None:
        obj = self._get_resource(_obj.Object, obj)
        if obj.container is not None:
            return obj.container
    if container is not None:
        container = self._get_resource(_container.Container, container)
        return container.name
    raise ValueError('container must be specified')