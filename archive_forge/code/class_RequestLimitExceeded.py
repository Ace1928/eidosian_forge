import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class RequestLimitExceeded(HeatException):
    msg_fmt = _('Request limit exceeded: %(message)s')