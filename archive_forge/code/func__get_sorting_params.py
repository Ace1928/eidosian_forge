import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def _get_sorting_params(self, params):
    """
        Process sorting params.
        Currently glance supports two sorting syntax: classic and new one,
        that is uniform for all OpenStack projects.
        Classic syntax: sort_key=name&sort_dir=asc&sort_key=size&sort_dir=desc
        New syntax: sort=name:asc,size:desc
        """
    sort_keys = []
    sort_dirs = []
    if 'sort' in params:
        if 'sort_key' in params or 'sort_dir' in params:
            msg = _('Old and new sorting syntax cannot be combined')
            raise webob.exc.HTTPBadRequest(explanation=msg)
        for sort_param in params.pop('sort').strip().split(','):
            key, _sep, dir = sort_param.partition(':')
            if not dir:
                dir = self._default_sort_dir
            sort_keys.append(self._validate_sort_key(key.strip()))
            sort_dirs.append(self._validate_sort_dir(dir.strip()))
    else:
        while 'sort_key' in params:
            sort_keys.append(self._validate_sort_key(params.pop('sort_key').strip()))
        while 'sort_dir' in params:
            sort_dirs.append(self._validate_sort_dir(params.pop('sort_dir').strip()))
        if sort_dirs:
            dir_len = len(sort_dirs)
            key_len = len(sort_keys)
            if dir_len > 1 and dir_len != key_len:
                msg = _('Number of sort dirs does not match the number of sort keys')
                raise webob.exc.HTTPBadRequest(explanation=msg)
    if not sort_keys:
        sort_keys = [self._default_sort_key]
    if not sort_dirs:
        sort_dirs = [self._default_sort_dir]
    return (sort_keys, sort_dirs)