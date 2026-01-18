from neutron_lib._i18n import _
from neutron_lib import exceptions
class DeviceDriverImportError(exceptions.NeutronException):
    message = _('Can not load driver :%(device_driver)s')