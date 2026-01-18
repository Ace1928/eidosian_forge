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
def _do_add(self, req, image, api_pol, change):
    path = change['path']
    path_root = path[0]
    value = change['value']
    json_schema_version = change.get('json_schema_version', 10)
    if path_root == 'locations':
        api_pol.update_locations()
        self._do_add_locations(image, path[1], value)
    else:
        api_pol.update_property(path_root, value)
        if (hasattr(image, path_root) or path_root in image.extra_properties) and json_schema_version == 4:
            msg = _('Property %s already present.')
            raise webob.exc.HTTPConflict(msg % path_root)
        if hasattr(image, path_root):
            setattr(image, path_root, value)
        else:
            image.extra_properties[path_root] = value