from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from urllib import parse as urlparse
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine.resources.openstack.heat import deployed_server
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _server_create_software_config_poll_heat(self, server_name='server_heat', md=None):
    stack_name = '%s_s' % server_name
    tmpl, stack = self._setup_test_stack(stack_name, server_heat_tmpl)
    self.stack = stack
    props = tmpl.t['resources']['server']['properties']
    props['software_config_transport'] = 'POLL_SERVER_HEAT'
    if md is not None:
        tmpl.t['resources']['server']['metadata'] = md
    self.server_props = props
    resource_defns = tmpl.resource_definitions(stack)
    server = deployed_server.DeployedServer('server', resource_defns['server'], stack)
    scheduler.TaskRunner(server.create)()
    self.assertEqual('1234', server._get_user_id())
    self.assertTrue(stack.access_allowed('1234', 'server'))
    self.assertFalse(stack.access_allowed('45678', 'server'))
    self.assertFalse(stack.access_allowed('4567', 'wserver'))
    return (stack, server)