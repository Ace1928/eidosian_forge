from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterIsNotExternal(exceptions.BadRequest):
    message = _('Router %(router_id)s has no external network gateway set')