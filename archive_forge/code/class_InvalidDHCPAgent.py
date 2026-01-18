from neutron_lib._i18n import _
from neutron_lib import exceptions
from neutron_lib.exceptions import agent as agent_exc
class InvalidDHCPAgent(agent_exc.AgentNotFound):
    message = _('Agent %(id)s is not a valid DHCP Agent or has been disabled.')