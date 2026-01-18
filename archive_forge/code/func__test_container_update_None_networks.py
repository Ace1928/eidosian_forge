import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_container_update_None_networks(self, new_networks):
    t = template_format.parse(zun_template_minimum)
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    rsrc_defn = resource_defns[self.fake_name]
    c = self._create_resource('container', rsrc_defn, stack)
    scheduler.TaskRunner(c.create)()
    new_t = copy.deepcopy(t)
    new_t['resources'][self.fake_name]['properties']['networks'] = new_networks
    rsrc_defns = template.Template(new_t).resource_definitions(stack)
    new_c = rsrc_defns[self.fake_name]
    iface = create_fake_iface(port='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', net='450abbc9-9b6d-4d6f-8c3a-c47ac34100ef', ip='1.2.3.4')
    self.client.containers.network_list.return_value = [iface]
    scheduler.TaskRunner(c.update, new_c)()
    self.assertEqual((c.UPDATE, c.COMPLETE), c.state)
    self.client.containers.network_list.assert_called_once_with(self.resource_id)