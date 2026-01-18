from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
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