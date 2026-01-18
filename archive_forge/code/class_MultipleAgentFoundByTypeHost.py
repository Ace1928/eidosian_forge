from neutron_lib._i18n import _
from neutron_lib import exceptions
class MultipleAgentFoundByTypeHost(exceptions.Conflict):
    message = _('Multiple agents with agent_type=%(agent_type)s and host=%(host)s found.')