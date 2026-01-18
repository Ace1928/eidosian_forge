from oslo_utils import excutils
from neutron_lib._i18n import _
class ProcessExecutionError(RuntimeError):

    def __init__(self, message, returncode):
        super().__init__(message)
        self.returncode = returncode