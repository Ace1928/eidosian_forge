import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class NotAuthorized(Forbidden):
    msg_fmt = _('You are not authorized to complete this action.')