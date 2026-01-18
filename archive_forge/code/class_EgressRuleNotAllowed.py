import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class EgressRuleNotAllowed(HeatException):
    msg_fmt = _("Egress rules are only allowed when Neutron is used and the 'VpcId' property is set.")