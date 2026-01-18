from .default import DefaultDeviceHandler
from warnings import warn
class CsrDeviceHandler(DefaultDeviceHandler):
    """
    Cisco CSR handler for device specific information.

    """

    def __init__(self, device_params):
        warn('CsrDeviceHandler is deprecated, please use IosxeDeviceHandler', DeprecationWarning, stacklevel=2)
        super(CsrDeviceHandler, self).__init__(device_params)

    def add_additional_ssh_connect_params(self, kwargs):
        warn('CsrDeviceHandler is deprecated, please use IosxeDeviceHandler', DeprecationWarning, stacklevel=2)
        kwargs['unknown_host_cb'] = csr_unknown_host_cb