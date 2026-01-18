import copy
import uuid
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources.aws.ec2 import network_interface as net_interfaces
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _create_test_instance_with_nic(self, return_server, name):
    stack_name = '%s_s' % name
    t = template_format.parse(wp_template_with_nic)
    kwargs = {'KeyName': 'test', 'InstanceType': 'm1.large', 'SubnetId': '4156c7a5-e8c4-4aff-a6e1-8f3c7bc83861'}
    tmpl = template.Template(t, env=environment.Environment(kwargs))
    self.stack = parser.Stack(utils.dummy_context(), stack_name, tmpl, stack_id=str(uuid.uuid4()))
    image_id = 'CentOS 5.2'
    t['Resources']['WebServer']['Properties']['ImageId'] = image_id
    resource_defns = self.stack.t.resource_definitions(self.stack)
    nic = net_interfaces.NetworkInterface('%s_nic' % name, resource_defns['nic1'], self.stack)
    instance = instances.Instance('%s_name' % name, resource_defns['WebServer'], self.stack)
    metadata = instance.metadata_get()
    self._mock_get_image_id_success(image_id, 1)
    self.stub_SubnetConstraint_validate()
    self.patchobject(nic, 'client', return_value=FakeNeutron())
    self.patchobject(neutron.NeutronClientPlugin, '_create', return_value=FakeNeutron())
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    server_userdata = instance.client_plugin().build_userdata(metadata, instance.properties['UserData'], 'ec2-user')
    self.patchobject(nova.NovaClientPlugin, 'build_userdata', return_value=server_userdata)
    self.patchobject(self.fc.servers, 'create', return_value=return_server)
    scheduler.TaskRunner(nic.create)()
    self.stack.resources['nic1'] = nic
    scheduler.TaskRunner(instance.create)()
    self.fc.servers.create.assert_called_once_with(image=1, flavor=3, key_name='test', name=utils.PhysName(stack_name, instance.name), security_groups=None, userdata=server_userdata, scheduler_hints=None, meta=None, nics=[{'port-id': '64d913c1-bcb1-42d2-8f0a-9593dbcaf251'}], availability_zone=None, block_device_mapping=None)
    self.m_f_i.assert_called_with(image_id)
    nova.NovaClientPlugin.build_userdata.assert_called_once_with(metadata, instance.properties['UserData'], 'ec2-user')
    neutron.NeutronClientPlugin._create.assert_called_once_with()
    nova.NovaClientPlugin.client.assert_called_with()
    glance.GlanceClientPlugin.find_image_by_name_or_id.assert_called_with(image_id)
    return instance