import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeTargetManagerSortingTest(VolumeTargetManagerTestBase):

    def setUp(self):
        super(VolumeTargetManagerSortingTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = ironicclient.v1.volume_target.VolumeTargetManager(self.api)

    def test_volume_targets_list_sort_key(self):
        volume_targets = self.mgr.list(sort_key='updated_at')
        expect = [('GET', '/v1/volume/targets/?sort_key=updated_at', {}, None)]
        expect_targets = [TARGET2, TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_sort_dir(self):
        volume_targets = self.mgr.list(sort_dir='desc')
        expect = [('GET', '/v1/volume/targets/?sort_dir=desc', {}, None)]
        expect_targets = [TARGET2, TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)