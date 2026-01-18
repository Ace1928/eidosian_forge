from neutron_lib._i18n import _
from neutron_lib import exceptions
class MixedIPVersionsForIPSecConnection(exceptions.BadRequest):
    message = _('IP versions are not compatible between peer and local endpoints')