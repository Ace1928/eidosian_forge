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
def _build_target_enforcement():
    target = {}
    try:
        target['credential'] = PROVIDERS.credential_api.get_credential(flask.request.view_args.get('credential_id'))
    except exception.NotFound:
        pass
    return target