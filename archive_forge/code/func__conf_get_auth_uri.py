import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
import requests
import webob
from heat.api.aws import exception
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
def _conf_get_auth_uri(self):
    auth_uri = self._conf_get('auth_uri')
    if auth_uri:
        return auth_uri.replace('v2.0', 'v3')
    else:
        return endpoint_utils.get_auth_uri()