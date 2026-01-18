from __future__ import absolute_import
from functools import wraps, partial
from flask import request, url_for, current_app
from flask import abort as original_flask_abort
from flask import make_response as original_flask_make_response
from flask.views import MethodView
from flask.signals import got_request_exception
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException, MethodNotAllowed, NotFound, NotAcceptable, InternalServerError
from werkzeug.wrappers import Response as ResponseBase
from flask_restful.utils import http_status_message, unpack, OrderedDict
from flask_restful.representations.json import output_json
import sys
from types import MethodType
import operator
def _get_propagate_exceptions_bool(app):
    """Handle Flask's propagate_exceptions.

    If propagate_exceptions is set to True then the exceptions are re-raised rather than being handled
    by the app’s error handlers.

    The default value for Flask's app.config['PROPAGATE_EXCEPTIONS'] is None. In this case return a sensible
    value: self.testing or self.debug.
    """
    propagate_exceptions = app.config.get(_PROPAGATE_EXCEPTIONS, False)
    if propagate_exceptions is None:
        return app.testing or app.debug
    return propagate_exceptions