import random
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_utils import timeutils
def _wait_secs(self, action):
    """Return a (randomised) wait time for the specified action."""
    if action not in self.properties[self.DELAY_ACTIONS]:
        return 0
    min_wait_secs, max_jitter_secs = self._delay_parameters()
    return min_wait_secs + max_jitter_secs * random.random()