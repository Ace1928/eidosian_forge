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
class OAuth1ListAccessTokensResource(_OAuth1ResourceBase):

    def get(self, user_id):
        """List OAuth1 Access Tokens for user.

        GET /v3/users/{user_id}/OS-OAUTH1/access_tokens
        """
        ENFORCER.enforce_call(action='identity:list_access_tokens')
        if self.oslo_context.is_delegated_auth:
            raise ks_exception.Forbidden(_('Cannot list request tokens with a token issued via delegation.'))
        refs = PROVIDERS.oauth_api.list_access_tokens(user_id)
        formatted_refs = [_format_token_entity(x) for x in refs]
        return self.wrap_collection(formatted_refs)