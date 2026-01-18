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
def _deferred_blueprint_init(self, setup_state):
    """Synchronize prefix between blueprint/api and registration options, then
        perform initialization with setup_state.app :class:`flask.Flask` object.
        When a :class:`flask_restful.Api` object is initialized with a blueprint,
        this method is recorded on the blueprint to be run when the blueprint is later
        registered to a :class:`flask.Flask` object.  This method also monkeypatches
        BlueprintSetupState.add_url_rule with _blueprint_setup_add_url_rule_patch.

        :param setup_state: The setup state object passed to deferred functions
            during blueprint registration
        :type setup_state: flask.blueprints.BlueprintSetupState

        """
    self.blueprint_setup = setup_state
    if setup_state.add_url_rule.__name__ != '_blueprint_setup_add_url_rule_patch':
        setup_state._original_add_url_rule = setup_state.add_url_rule
        setup_state.add_url_rule = MethodType(Api._blueprint_setup_add_url_rule_patch, setup_state)
    if not setup_state.first_registration:
        raise ValueError('flask-restful blueprints can only be registered once.')
    self._init_app(setup_state.app)