import flask
from oslo_utils import timeutils
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
from keystone.oauth1 import core as oauth
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
class OAuth(base.AuthMethodHandler):

    def authenticate(self, auth_payload):
        """Turn a signed request with an access key into a keystone token."""
        response_data = {}
        oauth_headers = oauth.get_oauth_headers(flask.request.headers)
        access_token_id = oauth_headers.get('oauth_token')
        if not access_token_id:
            raise exception.ValidationError(attribute='oauth_token', target='request')
        acc_token = PROVIDERS.oauth_api.get_access_token(access_token_id)
        expires_at = acc_token['expires_at']
        if expires_at:
            now = timeutils.utcnow()
            expires = timeutils.normalize_time(timeutils.parse_isotime(expires_at))
            if now > expires:
                raise exception.Unauthorized(_('Access token is expired'))
        url = ks_flask.base_url(path=flask.request.path)
        access_verifier = oauth.ResourceEndpoint(request_validator=validator.OAuthValidator(), token_generator=oauth.token_generator)
        result, request = access_verifier.validate_protected_resource_request(url, http_method='POST', body=flask.request.args, headers=dict(flask.request.headers), realms=None)
        if not result:
            msg = _('Could not validate the access token')
            raise exception.Unauthorized(msg)
        response_data['user_id'] = acc_token['authorizing_user_id']
        response_data['access_token_id'] = access_token_id
        response_data['project_id'] = acc_token['project_id']
        return base.AuthHandlerResponse(status=True, response_body=None, response_data=response_data)