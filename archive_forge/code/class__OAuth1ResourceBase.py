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
class _OAuth1ResourceBase(ks_flask.ResourceBase):
    collection_key = 'access_tokens'
    member_key = 'access_token'

    @classmethod
    def _add_self_referential_link(cls, ref, collection_name=None):
        ref.setdefault('links', {})
        path = '/users/%(user_id)s/OS-OAUTH1/access_tokens' % {'user_id': ref.get('authorizing_user_id', '')}
        ref['links']['self'] = ks_flask.base_url(path) + '/' + ref['id']