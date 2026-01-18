from openstack.clustering.v1 import build_info
from openstack.tests.unit import base
class TestBuildInfo(base.TestCase):

    def setUp(self):
        super(TestBuildInfo, self).setUp()

    def test_basic(self):
        sot = build_info.BuildInfo()
        self.assertEqual('/build-info', sot.base_path)
        self.assertEqual('build_info', sot.resource_key)
        self.assertTrue(sot.allow_fetch)

    def test_instantiate(self):
        sot = build_info.BuildInfo(**FAKE)
        self.assertEqual(FAKE['api'], sot.api)
        self.assertEqual(FAKE['engine'], sot.engine)