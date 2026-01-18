from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class NoKeyTest(common.HeatTestCase):

    def test_nokey_create(self):
        stack_name = 's_nokey'
        t = template_format.parse(nokey_template)
        stack = utils.parse_stack(t, stack_name=stack_name)
        resource_defns = stack.t.resource_definitions(stack)
        instance = instances.Instance('create_instance_name', resource_defns['WebServer'], stack)
        metadata = instance.metadata_get()
        server_userdata = instance.client_plugin().build_userdata(metadata, instance.properties['UserData'], 'ec2-user')
        fc = fakes_nova.FakeClient()
        self.patchobject(nova.NovaClientPlugin, 'client', return_value=fc)
        self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=1234)
        self.patchobject(nova.NovaClientPlugin, 'build_userdata', return_value=server_userdata)
        self.patchobject(fc.servers, 'create', return_value=fc.servers.list()[1])
        scheduler.TaskRunner(instance.create)()
        fc.servers.create.assert_called_once_with(image=1234, flavor=3, key_name=None, name=utils.PhysName(stack_name, instance.name), security_groups=None, userdata=server_userdata, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)