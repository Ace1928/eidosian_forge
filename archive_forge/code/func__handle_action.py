import random
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_utils import timeutils
def _handle_action(self):
    """Return a tuple of the start time in UTC and the time to wait."""
    return (timeutils.utcnow(), self._wait_secs(self.action))