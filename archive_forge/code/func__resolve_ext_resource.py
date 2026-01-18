from unittest import mock
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron.sfc import port_pair_group
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _resolve_ext_resource(self):
    value = mock.MagicMock()
    value.id = '[port1]'
    return value.id