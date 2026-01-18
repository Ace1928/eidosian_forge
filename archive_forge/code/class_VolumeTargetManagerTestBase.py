import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeTargetManagerTestBase(testtools.TestCase):

    def _validate_obj(self, expect, obj):
        self.assertEqual(expect['uuid'], obj.uuid)
        self.assertEqual(expect['volume_type'], obj.volume_type)
        self.assertEqual(expect['boot_index'], obj.boot_index)
        self.assertEqual(expect['volume_id'], obj.volume_id)
        self.assertEqual(expect['node_uuid'], obj.node_uuid)

    def _validate_list(self, expect_request, expect_targets, actual_targets):
        self.assertEqual(expect_request, self.api.calls)
        self.assertEqual(len(expect_targets), len(actual_targets))
        for expect, obj in zip(expect_targets, actual_targets):
            self._validate_obj(expect, obj)