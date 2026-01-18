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
def _test_instance_create_delete(self, vm_status='ACTIVE', vm_delete_status='NotFound'):
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance(return_server, 'in_cr_del')
    instance.resource_id = '1234'
    instance.status = vm_status
    self.assertGreater(instance.id, 0)
    d1 = {'server': self.fc.client.get_servers_detail()[1]['servers'][0]}
    d1['server']['status'] = vm_status
    mock_get = mock.Mock()
    self.fc.client.get_servers_1234 = mock_get
    d2 = copy.deepcopy(d1)
    if vm_delete_status == 'DELETED':
        d2['server']['status'] = vm_delete_status
        mock_get.side_effect = [(200, d1), (200, d2)]
    else:
        mock_get.side_effect = [(200, d1), fakes_nova.fake_exception()]
    scheduler.TaskRunner(instance.delete)()
    self.assertEqual((instance.DELETE, instance.COMPLETE), instance.state)
    self.assertEqual(2, mock_get.call_count)