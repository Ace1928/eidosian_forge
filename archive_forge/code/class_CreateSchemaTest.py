import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
class CreateSchemaTest(utils.BaseTestCase):

    def test_schema(self):
        schema = create_resources._CREATE_SCHEMA
        jsonschema.validate(valid_json, schema)
        jsonschema.validate(ironic_pov_invalid_json, schema)
        self.assertRaises(jsonschema.ValidationError, jsonschema.validate, schema_pov_invalid_json, schema)