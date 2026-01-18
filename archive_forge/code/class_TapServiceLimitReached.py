from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapServiceLimitReached(qexception.OverQuota):
    message = _('Reached the maximum quota for Tap Services')