import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
class WaitConditionFailure(exception.Error):

    def __init__(self, wait_condition, handle):
        reasons = handle.get_status_reason(handle.STATUS_FAILURE)
        super(WaitConditionFailure, self).__init__(';'.join(reasons))