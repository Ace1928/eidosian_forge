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
def _validate_import_body(self, body):
    try:
        method = body['method']
    except KeyError:
        msg = _("Import request requires a 'method' field.")
        raise webob.exc.HTTPBadRequest(explanation=msg)
    try:
        method_name = method['name']
    except KeyError:
        msg = _("Import request requires a 'name' field.")
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if method_name not in CONF.enabled_import_methods:
        msg = _("Unknown import method name '%s'.") % method_name
        raise webob.exc.HTTPBadRequest(explanation=msg)
    all_stores_must_succeed = body.get('all_stores_must_succeed', True)
    if not isinstance(all_stores_must_succeed, bool):
        msg = _("'all_stores_must_succeed' must be boolean value only")
        raise webob.exc.HTTPBadRequest(explanation=msg)
    all_stores = body.get('all_stores', False)
    if not isinstance(all_stores, bool):
        msg = _("'all_stores' must be boolean value only")
        raise webob.exc.HTTPBadRequest(explanation=msg)