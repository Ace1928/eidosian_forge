import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class NotSupported(HeatException):
    msg_fmt = _('%(feature)s is not supported.')