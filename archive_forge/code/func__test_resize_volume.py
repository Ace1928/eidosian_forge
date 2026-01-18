from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def _test_resize_volume(self, instance, id):
    self._set_action_mock()
    self.instances.resize_volume(instance, 1024)
    self.assertEqual(id, self._instance_id)
    self.assertEqual({'resize': {'volume': {'size': 1024}}}, self._body)