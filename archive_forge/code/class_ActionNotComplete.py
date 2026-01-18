import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ActionNotComplete(HeatException):
    msg_fmt = _('Stack %(stack_name)s has an action (%(action)s) in progress or failed state.')