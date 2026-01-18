from neutron_lib._i18n import _
from neutron_lib import exceptions
class InvalidPTRZoneConfiguration(exceptions.Conflict):
    message = _('Value of %(parameter)s has to be multiple of %(number)s, with maximum value of %(maximum)s and minimum value of %(minimum)s')