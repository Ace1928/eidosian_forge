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
def _create_test_instance(self, return_server, name):
    instance = self._setup_test_instance(return_server, name)
    bdm = {'vdb': '9ef5496e-7426-446a-bbc8-01f84d9c9972:snap::True'}
    mock_get = mock.Mock(return_value=return_server)
    with mock.patch.object(self.fc.servers, 'get', mock_get):
        scheduler.TaskRunner(instance.create)()
    self.mock_create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(self.stack.name, instance.name, limit=instance.physical_resource_name_limit), security_groups=None, userdata=mock.ANY, scheduler_hints={'foo': ['spam', 'ham', 'baz'], 'bar': 'eggs'}, meta=None, nics=None, availability_zone=None, block_device_mapping=bdm)
    mock_get.assert_called_with(return_server.id)
    return instance