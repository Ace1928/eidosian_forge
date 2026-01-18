from octavia_lib.i18n import _
class DriverAgentNotFound(Exception):
    """Exception raised when the driver agent cannot be reached.

    Each exception will include a message field that describes the
    error.
    :param fault_string: String describing the fault.
    :type fault_string: string
    """
    fault_string = _('The driver-agent process was not found or not ready.')

    def __init__(self, *args, **kwargs):
        self.fault_string = kwargs.pop('fault_string', self.fault_string)
        super().__init__(self.fault_string, *args, **kwargs)