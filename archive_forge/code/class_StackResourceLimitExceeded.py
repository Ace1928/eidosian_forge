import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class StackResourceLimitExceeded(HeatException):
    msg_fmt = _('Maximum resources per stack exceeded.')