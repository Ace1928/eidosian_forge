from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class StructuredDeploymentWithStrictInputTest(common.HeatTestCase):
    template = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'deploy_mysql': {'Type': 'OS::Heat::StructuredDeployment', 'Properties': {}}}}

    def setUp(self):
        super(StructuredDeploymentWithStrictInputTest, self).setUp()
        self.source = {'config': {'foo': [{'get_input': 'bar'}, {'get_input': 'barz'}]}}
        self.inputs = [swc_io.InputConfig(name='bar', value='baz'), swc_io.InputConfig(name='barz', value='baz2')]

    def _stack_with_template(self, template_def):
        self.ctx = utils.dummy_context()
        self.stack = parser.Stack(self.ctx, 'software_deploly_test_stack', template.Template(template_def))
        self.deployment = self.stack['deploy_mysql']

    def test_build_derived_config_failure(self):
        props = {'input_values_validate': 'STRICT'}
        self.template['Resources']['deploy_mysql']['Properties'] = props
        self._stack_with_template(self.template)
        self.assertRaises(exception.UserParameterMissing, self.deployment._build_derived_config, 'CREATE', self.source, self.inputs[:1], {})

    def test_build_derived_config_success(self):
        props = {'input_values_validate': 'STRICT'}
        self.template['Resources']['deploy_mysql']['Properties'] = props
        self._stack_with_template(self.template)
        expected = {'foo': ['baz', 'baz2']}
        result = self.deployment._build_derived_config('CREATE', self.source, self.inputs, {})
        self.assertEqual(expected, result)