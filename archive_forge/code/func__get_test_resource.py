import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _get_test_resource(self, template):
    self.stack = utils.parse_stack(template)
    definition = self.stack.t.resource_definitions(self.stack)['kp']
    kp_res = keypair.KeyPair('kp', definition, self.stack)
    return kp_res