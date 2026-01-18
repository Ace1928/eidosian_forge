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
def _format_token_entity(entity):
    formatted_entity = entity.copy()
    access_token_id = formatted_entity['id']
    user_id = formatted_entity.get('authorizing_user_id', '')
    if 'role_ids' in entity:
        formatted_entity.pop('role_ids')
    if 'access_secret' in entity:
        formatted_entity.pop('access_secret')
    url = '/users/%(user_id)s/OS-OAUTH1/access_tokens/%(access_token_id)s/roles' % {'user_id': user_id, 'access_token_id': access_token_id}
    formatted_entity.setdefault('links', {})
    formatted_entity['links']['roles'] = ks_flask.base_url(url)
    return formatted_entity