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
def _handle_unknown_keystone_exception(error):
    if isinstance(error, TypeError):
        new_exc = exception.ValidationError(error)
    else:
        new_exc = exception.UnexpectedError(error)
    return _handle_keystone_exception(new_exc)