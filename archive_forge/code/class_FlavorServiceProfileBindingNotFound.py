from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorServiceProfileBindingNotFound(exceptions.NotFound):
    message = _('Service Profile %(sp_id)s is not associated with flavor %(fl_id)s.')