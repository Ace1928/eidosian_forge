import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class UnableToAutoAllocateNetwork(HeatException):
    msg_fmt = _('Unable to automatically allocate a network: %(message)s')