from neutron_lib._i18n import _
from neutron_lib import exceptions
class HAMaximumAgentsNumberNotValid(exceptions.NeutronException):
    message = _('max_l3_agents_per_router %(max_agents)s config parameter is not valid as it cannot be negative. It must be 1 or greater. Alternatively, it can be 0 to mean unlimited.')