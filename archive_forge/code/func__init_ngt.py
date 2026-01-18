from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_ngt(self, template):
    self.stack = utils.parse_stack(template)
    return self.stack['node-group']