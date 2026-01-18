import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def _validate_node_volume_target_list(self, expect, volume_targets):
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(volume_targets))
    self.assertIsInstance(volume_targets[0], volume_target.VolumeTarget)
    self.assertEqual(TARGET['uuid'], volume_targets[0].uuid)
    self.assertEqual(TARGET['volume_type'], volume_targets[0].volume_type)
    self.assertEqual(TARGET['boot_index'], volume_targets[0].boot_index)
    self.assertEqual(TARGET['volume_id'], volume_targets[0].volume_id)
    self.assertEqual(TARGET['node_uuid'], volume_targets[0].node_uuid)