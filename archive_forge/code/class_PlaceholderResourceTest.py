from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class PlaceholderResourceTest(common.HeatTestCase):
    tmpl = '\nheat_template_version: 2015-10-15\nresources:\n  none:\n    type: OS::BAR::FOO\n    properties:\n      ignored: foo\n'

    class FooResource(none.NoneResource):
        default_client_name = 'heat'
        entity = 'foo'
    FOO_RESOURCE_TYPE = 'OS::BAR::FOO'

    def setUp(self):
        super(PlaceholderResourceTest, self).setUp()
        resource._register_class(self.FOO_RESOURCE_TYPE, self.FooResource)
        self.t = template_format.parse(self.tmpl)
        self.stack = utils.parse_stack(self.t)
        self.rsrc = self.stack['none']
        self.client = mock.MagicMock()
        self.patchobject(self.FooResource, 'client', return_value=self.client)
        scheduler.TaskRunner(self.rsrc.create)()

    def _test_delete(self, is_placeholder=True):
        if not is_placeholder:
            delete_call_count = 1
            self.rsrc.data = mock.Mock(return_value={})
        else:
            delete_call_count = 0
            self.rsrc.data = mock.Mock(return_value={'is_placeholder': 'True'})
        scheduler.TaskRunner(self.rsrc.delete)()
        self.assertEqual((self.rsrc.DELETE, self.rsrc.COMPLETE), self.rsrc.state)
        self.assertEqual(delete_call_count, self.client.foo.delete.call_count)
        self.assertEqual('foo', self.rsrc.entity)

    def test_not_placeholder_resource_delete(self):
        self._test_delete(is_placeholder=False)

    def test_placeholder_resource_delete(self):
        self._test_delete()