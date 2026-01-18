from openstack.accelerator.v2 import _proxy
from openstack.accelerator.v2 import accelerator_request
from openstack.accelerator.v2 import deployable
from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestAcceleratorDevice(TestAcceleratorProxy):

    def test_list_device_profile(self):
        self.verify_list(self.proxy.device_profiles, device_profile.DeviceProfile)

    def test_create_device_profile(self):
        self.verify_create(self.proxy.create_device_profile, device_profile.DeviceProfile)

    def test_delete_device_profile(self):
        self.verify_delete(self.proxy.delete_device_profile, device_profile.DeviceProfile, False)

    def test_delete_device_profile_ignore(self):
        self.verify_delete(self.proxy.delete_device_profile, device_profile.DeviceProfile, True)

    def test_get_device_profile(self):
        self.verify_get(self.proxy.get_device_profile, device_profile.DeviceProfile)