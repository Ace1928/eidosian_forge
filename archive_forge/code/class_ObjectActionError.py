import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ObjectActionError(HeatException):
    msg_fmt = _('Object action %(action)s failed because: %(reason)s')