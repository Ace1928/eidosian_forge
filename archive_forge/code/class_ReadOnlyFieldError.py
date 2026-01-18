import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ReadOnlyFieldError(HeatException):
    msg_fmt = _('Cannot modify readonly field %(field)s')