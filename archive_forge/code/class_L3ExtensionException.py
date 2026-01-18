from neutron_lib._i18n import _
from neutron_lib import exceptions
class L3ExtensionException(exceptions.NeutronException):
    message = _('The following L3 agent extensions do not inherit from ``neutron_lib.agent.l3_extension.L3AgentExtension``: %(extensions)s.')