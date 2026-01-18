from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosRuleNotSupportedByNetwork(e.Conflict):
    message = _('Rule %(rule_type)s is not supported by network %(network_id)s')