import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
def _convert_v3_to_ec2_credential(credential):
    try:
        blob = jsonutils.loads(credential['blob'])
    except TypeError:
        blob = credential['blob']
    return {'user_id': credential.get('user_id'), 'tenant_id': credential.get('project_id'), 'access': blob.get('access'), 'secret': blob.get('secret'), 'trust_id': blob.get('trust_id')}