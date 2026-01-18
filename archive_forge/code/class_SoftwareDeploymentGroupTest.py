import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SoftwareDeploymentGroupTest(common.HeatTestCase):
    template = {'heat_template_version': '2013-05-23', 'resources': {'deploy_mysql': {'type': 'OS::Heat::SoftwareDeploymentGroup', 'properties': {'config': 'config_uuid', 'servers': {'server1': 'uuid1', 'server2': 'uuid2'}, 'input_values': {'foo': 'bar'}, 'name': '10_config'}}}}

    def setUp(self):
        common.HeatTestCase.setUp(self)
        self.rpc_client = mock.MagicMock()

    def test_build_resource_definition(self):
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sd.SoftwareDeploymentGroup('test', snip, stack)
        expect = rsrc_defn.ResourceDefinition(None, 'OS::Heat::SoftwareDeployment', {'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'server': 'uuid1', 'signal_transport': 'CFN_SIGNAL'})
        rdef = resg.get_resource_def()
        self.assertEqual(expect, resg.build_resource_definition('server1', rdef))
        rdef = resg.get_resource_def(include_all=True)
        self.assertEqual(expect, resg.build_resource_definition('server1', rdef))

    def test_resource_names(self):
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sd.SoftwareDeploymentGroup('test', snip, stack)
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
        resg = sd.SoftwareDeploymentGroup('test', snip, stack)
        templ = {'heat_template_version': '2015-04-30', 'resources': {'server1': {'type': 'OS::Heat::SoftwareDeployment', 'properties': {'server': 'uuid1', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'signal_transport': 'CFN_SIGNAL'}}, 'server2': {'type': 'OS::Heat::SoftwareDeployment', 'properties': {'server': 'uuid2', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'signal_transport': 'CFN_SIGNAL'}}}}
        self.assertEqual(templ, resg._assemble_nested(['server1', 'server2']).t)

    def test_validate(self):
        stack = utils.parse_stack(self.template)
        snip = stack.t.resource_definitions(stack)['deploy_mysql']
        resg = sd.SoftwareDeploymentGroup('deploy_mysql', snip, stack)
        self.assertIsNone(resg.validate())