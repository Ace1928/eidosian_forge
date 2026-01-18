from .default import DefaultDeviceHandler
class IosxrDeviceHandler(DefaultDeviceHandler):
    """
    Cisco IOS-XR handler for device specific information.

    """

    def __init__(self, device_params):
        super(IosxrDeviceHandler, self).__init__(device_params)

    def add_additional_ssh_connect_params(self, kwargs):
        kwargs['unknown_host_cb'] = iosxr_unknown_host_cb

    def perform_qualify_check(self):
        return False