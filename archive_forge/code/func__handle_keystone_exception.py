import functools
import sys
import flask
import oslo_i18n
from oslo_log import log
from oslo_middleware import healthcheck
import keystone.api
from keystone import exception
from keystone.oauth2 import handlers as oauth2_handlers
from keystone.receipt import handlers as receipt_handlers
from keystone.server.flask import common as ks_flask
from keystone.server.flask.request_processing import json_body
from keystone.server.flask.request_processing import req_logging
def _handle_keystone_exception(error):
    if isinstance(error, exception.InsufficientAuthMethods):
        return receipt_handlers.build_receipt(error)
    elif isinstance(error, exception.OAuth2Error):
        return oauth2_handlers.build_response(error)
    if isinstance(error, exception.RedirectRequired):
        return flask.redirect(error.redirect_url)
    if isinstance(error, exception.Unauthorized):
        LOG.warning('Authorization failed. %(exception)s from %(remote_addr)s', {'exception': error, 'remote_addr': flask.request.remote_addr})
    else:
        LOG.exception(str(error))
    error_message = error.args[0]
    message = oslo_i18n.translate(error_message, _best_match_language())
    if message is error_message:
        message = str(message)
    body = dict(error={'code': error.code, 'title': error.title, 'message': message})
    if isinstance(error, exception.AuthPluginException):
        body['error']['identity'] = error.authentication
    response = flask.jsonify(body)
    response.status_code = error.code
    if isinstance(error, exception.Unauthorized):
        url = ks_flask.base_url()
        response.headers['WWW-Authenticate'] = 'Keystone uri="%s"' % url
    return response