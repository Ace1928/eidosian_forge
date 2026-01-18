from neutron_lib._i18n import _
from neutron_lib import exceptions
class AvailabilityZoneNotFound(exceptions.NotFound):
    message = _('AvailabilityZone %(availability_zone)s could not be found.')