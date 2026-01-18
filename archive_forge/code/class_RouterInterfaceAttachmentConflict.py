from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterInterfaceAttachmentConflict(exceptions.Conflict):
    message = _('Error %(reason)s while attempting the operation.')