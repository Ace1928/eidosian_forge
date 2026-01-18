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
@staticmethod
def _blob_to_json(ref):
    blob = ref.get('blob')
    if isinstance(blob, dict):
        ref = ref.copy()
        ref['blob'] = jsonutils.dumps(blob)
    return ref