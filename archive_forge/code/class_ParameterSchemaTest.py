from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParameterSchemaTest(common.HeatTestCase):

    def test_validate_schema_wrong_key(self):
        error = self.assertRaises(exception.InvalidSchemaError, parameters.Schema.from_dict, 'param_name', {'foo': 'bar'})
        self.assertEqual("Invalid key 'foo' for parameter (param_name)", str(error))

    def test_validate_schema_no_type(self):
        error = self.assertRaises(exception.InvalidSchemaError, parameters.Schema.from_dict, 'broken', {'Description': 'Hi!'})
        self.assertEqual('Missing parameter type for parameter: broken', str(error))