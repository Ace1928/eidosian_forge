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
class StructuredDeploymentGroupTest(common.HeatTestCase):
    template = {'heat_template_version': '2013-05-23', 'resources': {'deploy_mysql': {'type': 'OS::Heat::StructuredDeploymentGroup', 'properties': {'config': 'config_uuid', 'servers': {'server1': 'uuid1', 'server2': 'uuid2'}}}}}

    def test_build_resource_definition(self):
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sc.StructuredDeploymentGroup('test', snip, stack)
        expect = rsrc_defn.ResourceDefinition(None, 'OS::Heat::StructuredDeployment', {'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': None, 'name': None, 'server': 'uuid1', 'input_key': 'get_input', 'signal_transport': 'CFN_SIGNAL', 'input_values_validate': 'LAX'})
        rdef = resg.get_resource_def()
        self.assertEqual(expect, resg.build_resource_definition('server1', rdef))
        rdef = resg.get_resource_def(include_all=True)
        self.assertEqual(expect, resg.build_resource_definition('server1', rdef))

    def test_resource_names(self):
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sc.StructuredDeploymentGroup('test', snip, stack)
        self.assertEqual(set(('server1', 'server2')), set(resg._resource_names()))
        resg.properties = {'servers': {'s1': 'u1', 's2': 'u2', 's3': 'u3'}}
        self.assertEqual(set(('s1', 's2', 's3')), set(resg._resource_names()))

    def test_assemble_nested(self):
        """Tests nested stack implements group creation based on properties.

        Tests that the nested stack that implements the group is created
        appropriately based on properties.
        """
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sc.StructuredDeploymentGroup('test', snip, stack)
        templ = {'heat_template_version': '2015-04-30', 'resources': {'server1': {'type': 'OS::Heat::StructuredDeployment', 'properties': {'server': 'uuid1', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_key': 'get_input', 'input_values': None, 'name': None, 'signal_transport': 'CFN_SIGNAL', 'input_values_validate': 'LAX'}}, 'server2': {'type': 'OS::Heat::StructuredDeployment', 'properties': {'server': 'uuid2', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_key': 'get_input', 'input_values': None, 'name': None, 'signal_transport': 'CFN_SIGNAL', 'input_values_validate': 'LAX'}}}}
        self.assertEqual(templ, resg._assemble_nested(['server1', 'server2']).t)