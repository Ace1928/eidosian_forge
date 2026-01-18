from neutron_lib._i18n import _
from neutron_lib import exceptions
class AgentNotFoundByTypeHost(exceptions.NotFound):
    message = _('Agent with agent_type=%(agent_type)s and host=%(host)s could not be found.')