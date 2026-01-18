from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapServiceNotFound(qexception.NotFound):
    message = _('Tap Service  %(tap_id)s does not exist')