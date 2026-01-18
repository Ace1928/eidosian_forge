import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
class LimitModelTestCase(test_v3.RestfulTestCase):

    def test_get_default_limit_model_response_schema(self):
        schema = {'type': 'object', 'properties': {'model': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'description': {'type': 'string'}}, 'required': ['name', 'description'], 'additionalProperties': False}}, 'required': ['model'], 'additionalProperties': False}
        validator = validators.SchemaValidator(schema)
        response = self.get('/limits/model')
        validator.validate(response.json_body)

    def test_head_limit_model(self):
        self.head('/limits/model', expected_status=http.client.OK)

    def test_get_limit_model_returns_default_model(self):
        response = self.get('/limits/model')
        model = response.result
        expected = {'model': {'name': 'flat', 'description': 'Limit enforcement and validation does not take project hierarchy into consideration.'}}
        self.assertDictEqual(expected, model)

    def test_get_limit_model_without_token_fails(self):
        self.get('/limits/model', noauth=True, expected_status=http.client.UNAUTHORIZED)

    def test_head_limit_model_without_token_fails(self):
        self.head('/limits/model', noauth=True, expected_status=http.client.UNAUTHORIZED)