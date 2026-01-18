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
def _do_remove(self, req, image, api_pol, change):
    path = change['path']
    path_root = path[0]
    if path_root == 'locations':
        api_pol.delete_locations()
        try:
            self._do_remove_locations(image, path[1])
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(e.msg)
    else:
        api_pol.update_property(path_root)
        if hasattr(image, path_root):
            msg = _('Property %s may not be removed.')
            raise webob.exc.HTTPForbidden(msg % path_root)
        elif path_root in image.extra_properties:
            del image.extra_properties[path_root]
        else:
            msg = _('Property %s does not exist.')
            raise webob.exc.HTTPConflict(msg % path_root)