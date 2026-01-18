import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _test_instance_update_with_subnet(self, stack_name, new_interfaces=None, old_interfaces=None, need_update=True, multiple_get=True):
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    instance = self._create_test_instance(return_server, stack_name)
    self._stub_glance_for_update()
    iface = self.create_fake_iface('d1e9c73c-04fe-4e9e-983c-d5ef94cd1a46', 'c4485ba1-283a-4f5f-8868-0cd46cdda52f', '10.0.0.4')
    subnet_id = '8c1aaddf-e49e-4f28-93ea-ca9f0b3c6240'
    nics = [{'port-id': 'ea29f957-cd35-4364-98fb-57ce9732c10d'}]
    before_props = self.instance_props.copy()
    if old_interfaces is not None:
        before_props['NetworkInterfaces'] = old_interfaces
    update_props = copy.deepcopy(before_props)
    if new_interfaces is not None:
        update_props['NetworkInterfaces'] = new_interfaces
    update_props['SubnetId'] = subnet_id
    after = instance.t.freeze(properties=update_props)
    before = instance.t.freeze(properties=before_props)
    instance.reparse()
    self.fc.servers.get = mock.Mock(return_value=return_server)
    if need_update:
        return_server.interface_list = mock.Mock(return_value=[iface])
        return_server.interface_detach = mock.Mock(return_value=None)
        instance._build_nics = mock.Mock(return_value=nics)
        return_server.interface_attach = mock.Mock(return_value=None)
    scheduler.TaskRunner(instance.update, after, before)()
    self.assertEqual((instance.UPDATE, instance.COMPLETE), instance.state)
    if need_update:
        self.fc.servers.get.assert_called_with('1234')
        if not multiple_get:
            self.fc.servers.get.assert_called_once()
        return_server.interface_list.assert_called_once_with()
        return_server.interface_detach.assert_called_once_with('d1e9c73c-04fe-4e9e-983c-d5ef94cd1a46')
        instance._build_nics.assert_called_once_with(new_interfaces, security_groups=None, subnet_id=subnet_id)
        return_server.interface_attach.assert_called_once_with('ea29f957-cd35-4364-98fb-57ce9732c10d', None, None)
    else:
        self.fc.servers.get.assert_not_called()