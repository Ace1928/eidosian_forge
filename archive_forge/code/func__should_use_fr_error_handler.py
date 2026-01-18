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
def _should_use_fr_error_handler(self):
    """ Determine if error should be handled with FR or default Flask

        The goal is to return Flask error handlers for non-FR-related routes,
        and FR errors (with the correct media type) for FR endpoints. This
        method currently handles 404 and 405 errors.

        :return: bool
        """
    adapter = current_app.create_url_adapter(request)
    try:
        adapter.match()
    except MethodNotAllowed as e:
        valid_route_method = e.valid_methods[0]
        rule, _ = adapter.match(method=valid_route_method, return_rule=True)
        return self.owns_endpoint(rule.endpoint)
    except NotFound:
        return self.catch_all_404s
    except:
        pass