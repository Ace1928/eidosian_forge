import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
def _pre_process_event(self, event):
    current = self._current
    if current is None:
        raise excp.NotInitialized("Can not process event '%s'; the state machine hasn't been initialized" % event)
    if self._states[current.name]['terminal']:
        raise excp.InvalidState("Can not transition from terminal state '%s' on event '%s'" % (current.name, event))
    if event not in self._transitions[current.name]:
        raise excp.NotFound("Can not transition from state '%s' on event '%s' (no defined transition)" % (current.name, event))