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
def _setup_flask_restful_api(self):
    self.restful_api_url_prefix = '/_%s_TEST' % uuid.uuid4().hex
    self.restful_api = flask_restful.Api(self.public_app.app, self.restful_api_url_prefix)
    driver_simulation_method = self._driver_simulation_get_method

    class RestfulResource(flask_restful.Resource):

        def get(self, argument_id=None):
            if argument_id is not None:
                return self._get_argument(argument_id)
            return self._list_arguments()

        def _get_argument(self, argument_id):
            return {'argument': driver_simulation_method(argument_id)}

        def _list_arguments(self):
            return {'arguments': []}
    self.restful_api_resource = RestfulResource
    self.restful_api.add_resource(RestfulResource, '/argument/<string:argument_id>', '/argument')
    self.cleanup_instance('restful_api', 'restful_resource', 'restful_api_url_prefix')