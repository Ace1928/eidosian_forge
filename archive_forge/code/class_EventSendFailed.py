import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class EventSendFailed(HeatException):
    msg_fmt = _('Failed to send message to stack (%(stack_name)s) on other engine (%(engine_id)s)')