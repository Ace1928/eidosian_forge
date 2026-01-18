import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _method_not_allowed(self):
    """Raise a method not allowed error."""
    raise exception.OAuth2OtherError(int(http.client.METHOD_NOT_ALLOWED), http.client.responses[http.client.METHOD_NOT_ALLOWED], _('The method is not allowed for the requested URL.'))