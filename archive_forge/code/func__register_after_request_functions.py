import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _register_after_request_functions(self, functions=None):
    """Register functions to be executed in the `after request` phase.

        Override this method and pass in via "super" any additional functions
        that should be registered. It is assumed that any override will also
        accept a "functions" list and append the passed in values to it's
        list prior to calling super.

        Each function will be called with a single argument of the Response
        class type. The function must return either the passed in Response or
        a new Response. NOTE: As of flask 0.7, these functions may not be
        executed in the case of an unhandled exception.

        :param functions: list of functions that will be run in the
                          `after_request` phase.
        :type functions: list
        """
    functions = functions or []
    msg = 'after_request functions already registered'
    assert not self.__after_request_functions_added, msg
    self.__blueprint.after_request(_assert_rbac_enforcement_called)
    self.__blueprint.after_request(_remove_content_type_on_204)
    for f in functions:
        self.__blueprint.after_request(f)
    self.__after_request_functions_added = True