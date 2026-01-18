from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def _test_restart(self, instance, id):
    self._set_action_mock()
    self.instances.restart(instance)
    self.assertEqual(id, self._instance_id)
    self.assertEqual({'restart': {}}, self._body)