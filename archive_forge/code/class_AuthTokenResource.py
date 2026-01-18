import string
import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import strutils
import urllib
import werkzeug.exceptions
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.api._shared import saml
from keystone.auth import schema as auth_schema
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import utils as k_utils
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import schema as federation_schema
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
class AuthTokenResource(_AuthFederationWebSSOBase):

    def get(self):
        """Validate a token.

        HEAD/GET /v3/auth/tokens
        """
        if flask.request.method == 'HEAD':
            ENFORCER.enforce_call(action='identity:check_token')
        else:
            ENFORCER.enforce_call(action='identity:validate_token')
        token_id = flask.request.headers.get(authorization.SUBJECT_TOKEN_HEADER)
        access_rules_support = flask.request.headers.get(authorization.ACCESS_RULES_HEADER)
        allow_expired = strutils.bool_from_string(flask.request.args.get('allow_expired'))
        window_secs = CONF.token.allow_expired_window if allow_expired else 0
        include_catalog = 'nocatalog' not in flask.request.args
        token = PROVIDERS.token_provider_api.validate_token(token_id, window_seconds=window_secs, access_rules_support=access_rules_support)
        token_resp = render_token.render_token_response_from_model(token, include_catalog=include_catalog)
        resp_body = jsonutils.dumps(token_resp)
        response = flask.make_response(resp_body, http.client.OK)
        response.headers['X-Subject-Token'] = token_id
        response.headers['Content-Type'] = 'application/json'
        return response

    @ks_flask.unenforced_api
    def post(self):
        """Issue a token.

        POST /v3/auth/tokens
        """
        include_catalog = 'nocatalog' not in flask.request.args
        auth_data = self.request_body_json.get('auth')
        auth_schema.validate_issue_token_auth(auth_data)
        token = authentication.authenticate_for_token(auth_data)
        resp_data = render_token.render_token_response_from_model(token, include_catalog=include_catalog)
        resp_body = jsonutils.dumps(resp_data)
        response = flask.make_response(resp_body, http.client.CREATED)
        response.headers['X-Subject-Token'] = token.id
        response.headers['Content-Type'] = 'application/json'
        return response

    def delete(self):
        """Revoke a token.

        DELETE /v3/auth/tokens
        """
        ENFORCER.enforce_call(action='identity:revoke_token')
        token_id = flask.request.headers.get(authorization.SUBJECT_TOKEN_HEADER)
        PROVIDERS.token_provider_api.revoke_token(token_id)
        return (None, http.client.NO_CONTENT)