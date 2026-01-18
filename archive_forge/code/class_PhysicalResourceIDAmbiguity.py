import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class PhysicalResourceIDAmbiguity(HeatException):
    msg_fmt = _('Multiple resources were found with the physical ID (%(phys_id)s).')