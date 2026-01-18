from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPAssociationNotFound(exceptions.NotFound):
    message = _('Local IP %(local_ip_id)s association with port %(port_id)s could not be found.')