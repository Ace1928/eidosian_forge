from openstack.accelerator.v2 import _proxy
from openstack.accelerator.v2 import accelerator_request
from openstack.accelerator.v2 import deployable
from openstack.accelerator.v2 import device_profile
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestAcceleratorRequest(TestAcceleratorProxy):

    def test_list_accelerator_request(self):
        self.verify_list(self.proxy.accelerator_requests, accelerator_request.AcceleratorRequest)

    def test_create_accelerator_request(self):
        self.verify_create(self.proxy.create_accelerator_request, accelerator_request.AcceleratorRequest)

    def test_delete_accelerator_request(self):
        self.verify_delete(self.proxy.delete_accelerator_request, accelerator_request.AcceleratorRequest, False)

    def test_delete_accelerator_request_ignore(self):
        self.verify_delete(self.proxy.delete_accelerator_request, accelerator_request.AcceleratorRequest, True)

    def test_get_accelerator_request(self):
        self.verify_get(self.proxy.get_accelerator_request, accelerator_request.AcceleratorRequest)