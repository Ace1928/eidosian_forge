import copy
import http.client as http
import urllib.parse as urlparse
import debtcollector
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_serialization.jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _, _LW
import glance.notifier
import glance.schema
def _validate_create_body(self, body):
    """Validate the body of task creating request"""
    for param in self._required_properties:
        if param not in body:
            msg = _("Task '%s' is required") % param
            raise webob.exc.HTTPBadRequest(explanation=msg)