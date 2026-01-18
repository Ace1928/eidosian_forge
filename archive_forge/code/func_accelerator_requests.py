from openstack.accelerator.v2 import accelerator_request as _arq
from openstack.accelerator.v2 import deployable as _deployable
from openstack.accelerator.v2 import device as _device
from openstack.accelerator.v2 import device_profile as _device_profile
from openstack import proxy
def accelerator_requests(self, **query):
    """Retrieve a generator of accelerator requests.

        :param kwargs query: Optional query parameters to be sent to
            restrict the accelerator requests to be returned.
        :returns: A generator of accelerator request instances.
        """
    return self._list(_arq.AcceleratorRequest, **query)