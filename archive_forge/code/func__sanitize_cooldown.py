import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
def _sanitize_cooldown(self, cooldown):
    if cooldown is None:
        return 0
    return max(0, cooldown)