import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def _running(self):
    """Iterate over all subtasks that are currently running.

        Running subtasks are subtasks have been started but have not yet
        completed.
        """

    def running(k_r):
        return k_r[0] in self._graph and k_r[1].started()
    return filter(running, self._runners.items())