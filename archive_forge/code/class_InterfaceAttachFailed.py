import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class InterfaceAttachFailed(HeatException):
    msg_fmt = _('Failed to attach interface (%(port)s) to server (%(server)s)')