import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeTargetManagerPaginationTest(VolumeTargetManagerTestBase):

    def setUp(self):
        super(VolumeTargetManagerPaginationTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = ironicclient.v1.volume_target.VolumeTargetManager(self.api)

    def test_volume_targets_list_limit(self):
        volume_targets = self.mgr.list(limit=1)
        expect = [('GET', '/v1/volume/targets/?limit=1', {}, None)]
        expect_targets = [TARGET1]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_marker(self):
        volume_targets = self.mgr.list(marker=TARGET1['uuid'])
        expect = [('GET', '/v1/volume/targets/?marker=%s' % TARGET1['uuid'], {}, None)]
        expect_targets = [TARGET2]
        self._validate_list(expect, expect_targets, volume_targets)

    def test_volume_targets_list_pagination_no_limit(self):
        volume_targets = self.mgr.list(limit=0)
        expect = [('GET', '/v1/volume/targets', {}, None), ('GET', '/v1/volume/targets/?marker=%s' % TARGET1['uuid'], {}, None)]
        expect_targets = [TARGET1, TARGET2]
        self._validate_list(expect, expect_targets, volume_targets)