import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _client_secret_basic(self, client_id, client_secret):
    """Get an OAuth2.0 basic Access Token."""
    auth_data = {'identity': {'methods': ['application_credential'], 'application_credential': {'id': client_id, 'secret': client_secret}}}
    try:
        token = authentication.authenticate_for_token(auth_data)
    except exception.Error as error:
        if error.code == 401:
            error = exception.OAuth2InvalidClient(error.code, error.title, str(error))
        elif error.code == 400:
            error = exception.OAuth2InvalidRequest(error.code, error.title, str(error))
        else:
            error = exception.OAuth2OtherError(error.code, error.title, 'An unknown error occurred and failed to get an OAuth2.0 access token.')
        LOG.exception(error)
        raise error
    except Exception as error:
        error = exception.OAuth2OtherError(int(http.client.INTERNAL_SERVER_ERROR), http.client.responses[http.client.INTERNAL_SERVER_ERROR], str(error))
        LOG.exception(error)
        raise error
    resp = make_response({'access_token': token.id, 'token_type': 'Bearer', 'expires_in': CONF.token.expiration})
    resp.status = '200 OK'
    return resp