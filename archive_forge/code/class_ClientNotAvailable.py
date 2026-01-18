import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ClientNotAvailable(HeatException):
    msg_fmt = _('The client (%(client_name)s) is not available.')