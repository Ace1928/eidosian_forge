from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
class RestfulResource(flask_restful.Resource):

    def get(self, argument_id=None):
        if argument_id is not None:
            return self._get_argument(argument_id)
        return self._list_arguments()

    def _get_argument(self, argument_id):
        return {'argument': driver_simulation_method(argument_id)}

    def _list_arguments(self):
        return {'arguments': []}