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
@staticmethod
def _blueprint_setup_add_url_rule_patch(blueprint_setup, rule, endpoint=None, view_func=None, **options):
    """Method used to patch BlueprintSetupState.add_url_rule for setup
        state instance corresponding to this Api instance.  Exists primarily
        to enable _complete_url's function.

        :param blueprint_setup: The BlueprintSetupState instance (self)
        :param rule: A string or callable that takes a string and returns a
            string(_complete_url) that is the url rule for the endpoint
            being registered
        :param endpoint: See BlueprintSetupState.add_url_rule
        :param view_func: See BlueprintSetupState.add_url_rule
        :param **options: See BlueprintSetupState.add_url_rule
        """
    if callable(rule):
        rule = rule(blueprint_setup.url_prefix)
    elif blueprint_setup.url_prefix:
        rule = blueprint_setup.url_prefix + rule
    options.setdefault('subdomain', blueprint_setup.subdomain)
    if endpoint is None:
        endpoint = view_func.__name__
    defaults = blueprint_setup.url_defaults
    if 'defaults' in options:
        defaults = dict(defaults, **options.pop('defaults'))
    blueprint_setup.app.add_url_rule(rule, '%s.%s' % (blueprint_setup.blueprint.name, endpoint), view_func, defaults=defaults, **options)