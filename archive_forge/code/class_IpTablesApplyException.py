from neutron_lib._i18n import _
from neutron_lib import exceptions
class IpTablesApplyException(exceptions.NeutronException):

    def __init__(self, message=None):
        self.message = message
        super().__init__()