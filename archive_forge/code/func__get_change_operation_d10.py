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
def _get_change_operation_d10(self, raw_change):
    op = raw_change.get('op')
    if op is None:
        msg = _('Unable to find `op` in JSON Schema change. It must be one of the following: %(available)s.') % {'available': ', '.join(self._supported_operations)}
        raise webob.exc.HTTPBadRequest(explanation=msg)
    if op not in self._supported_operations:
        msg = _('Invalid operation: `%(op)s`. It must be one of the following: %(available)s.') % {'op': op, 'available': ', '.join(self._supported_operations)}
        raise webob.exc.HTTPBadRequest(explanation=msg)
    return op