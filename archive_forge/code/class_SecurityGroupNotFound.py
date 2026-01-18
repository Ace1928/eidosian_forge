from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
class SecurityGroupNotFound(exception.HeatException):
    msg_fmt = _('Security Group "%(group_name)s" not found')