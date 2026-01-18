from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients.os import swift
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class SwiftSignalTimeout(exception.Error):

    def __init__(self, wait_cond):
        reasons = wait_cond.get_status_reason(wait_cond.STATUS_SUCCESS)
        vals = {'len': len(reasons), 'count': wait_cond.properties[wait_cond.COUNT]}
        if reasons:
            vals['reasons'] = ';'.join(reasons)
            message = _('%(len)d of %(count)d received - %(reasons)s') % vals
        else:
            message = _('%(len)d of %(count)d received') % vals
        super(SwiftSignalTimeout, self).__init__(message)