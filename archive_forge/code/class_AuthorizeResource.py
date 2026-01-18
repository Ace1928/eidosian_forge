import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_utils import timeutils
from urllib import parse as urlparse
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.oauth1 import core as oauth1
from keystone.oauth1 import schema
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
class AuthorizeResource(_OAuth1ResourceBase):

    def put(self, request_token_id):
        ENFORCER.enforce_call(action='identity:authorize_request_token')
        roles = (flask.request.get_json(force=True, silent=True) or {}).get('roles', [])
        validation.lazy_validate(schema.request_token_authorize, roles)
        ctx = flask.request.environ[context.REQUEST_CONTEXT_ENV]
        if ctx.is_delegated_auth:
            raise exception.Forbidden(_('Cannot authorize a request token with a token issued via delegation.'))
        req_token = PROVIDERS.oauth_api.get_request_token(request_token_id)
        expires_at = req_token['expires_at']
        if expires_at:
            now = timeutils.utcnow()
            expires = timeutils.normalize_time(timeutils.parse_isotime(expires_at))
            if now > expires:
                raise exception.Unauthorized(_('Request token is expired'))
        authed_roles = _normalize_role_list(roles)
        try:
            auth_context = flask.request.environ[authorization.AUTH_CONTEXT_ENV]
            user_token_ref = auth_context['token']
        except KeyError:
            LOG.warning("Couldn't find the auth context.")
            raise exception.Unauthorized()
        user_id = user_token_ref.user_id
        project_id = req_token['requested_project_id']
        user_roles = PROVIDERS.assignment_api.get_roles_for_user_and_project(user_id, project_id)
        cred_set = set(user_roles)
        if not cred_set.issuperset(authed_roles):
            msg = _('authorizing user does not have role required')
            raise exception.Unauthorized(message=msg)
        role_ids = list(authed_roles)
        authed_token = PROVIDERS.oauth_api.authorize_request_token(request_token_id, user_id, role_ids)
        to_return = {'token': {'oauth_verifier': authed_token['verifier']}}
        return to_return