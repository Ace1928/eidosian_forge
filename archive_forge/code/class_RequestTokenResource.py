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
class RequestTokenResource(_OAuth1ResourceBase):

    @ks_flask.unenforced_api
    def post(self):
        oauth_headers = oauth1.get_oauth_headers(flask.request.headers)
        consumer_id = oauth_headers.get('oauth_consumer_key')
        requested_project_id = flask.request.headers.get('Requested-Project-Id')
        if not consumer_id:
            raise exception.ValidationError(attribute='oauth_consumer_key', target='request')
        if not requested_project_id:
            raise exception.ValidationError(attribute='Requested-Project-Id', target='request')
        PROVIDERS.resource_api.get_project(requested_project_id)
        PROVIDERS.oauth_api.get_consumer(consumer_id)
        url = _update_url_scheme()
        req_headers = {'Requested-Project-Id': requested_project_id}
        req_headers.update(flask.request.headers)
        request_verifier = oauth1.RequestTokenEndpoint(request_validator=validator.OAuthValidator(), token_generator=oauth1.token_generator)
        h, b, s = request_verifier.create_request_token_response(url, http_method='POST', body=flask.request.args, headers=req_headers)
        if not b:
            msg = _('Invalid signature')
            raise exception.Unauthorized(message=msg)
        oauth1.validate_oauth_params(b)
        request_token_duration = CONF.oauth1.request_token_duration
        token_ref = PROVIDERS.oauth_api.create_request_token(consumer_id, requested_project_id, request_token_duration, initiator=notifications.build_audit_initiator())
        result = 'oauth_token=%(key)s&oauth_token_secret=%(secret)s' % {'key': token_ref['id'], 'secret': token_ref['request_secret']}
        if CONF.oauth1.request_token_duration > 0:
            expiry_bit = '&oauth_expires_at=%s' % token_ref['expires_at']
            result += expiry_bit
        resp = flask.make_response(result, http.client.CREATED)
        resp.headers['Content-Type'] = 'application/x-www-form-urlencoded'
        return resp