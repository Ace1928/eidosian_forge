import hashlib
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.credential import schema
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _validate_blob_json(self, ref):
    try:
        blob = jsonutils.loads(ref.get('blob'))
    except (ValueError, TabError):
        raise exception.ValidationError(message=_('Invalid blob in credential'))
    if not blob or not isinstance(blob, dict):
        raise exception.ValidationError(attribute='blob', target='credential')
    if blob.get('access') is None:
        raise exception.ValidationError(attribute='access', target='credential')
    return blob