import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class IncompatibleObjectVersion(HeatException):
    msg_fmt = _('Version %(objver)s of %(objname)s is not supported')