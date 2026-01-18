from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import base
class TestDeviceProfile(base.TestCase):

    def test_basic(self):
        sot = device_profile.DeviceProfile()
        self.assertEqual('device_profile', sot.resource_key)
        self.assertEqual('device_profiles', sot.resources_key)
        self.assertEqual('/device_profiles', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_patch)

    def test_make_it(self):
        sot = device_profile.DeviceProfile(**FAKE)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['uuid'], sot.uuid)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['groups'], sot.groups)
        self.assertEqual(FAKE['description'], sot.description)