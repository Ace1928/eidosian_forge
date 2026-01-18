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
def _handle_flask_propagate_exceptions_config(app, e):
    propagate_exceptions = _get_propagate_exceptions_bool(app)
    if not isinstance(e, HTTPException) and propagate_exceptions:
        exc_type, exc_value, tb = sys.exc_info()
        if exc_value is e:
            raise
        else:
            raise e