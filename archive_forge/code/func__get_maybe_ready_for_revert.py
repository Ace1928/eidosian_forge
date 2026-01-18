import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def _get_maybe_ready_for_revert(self, atom):
    """Returns if an atom is *likely* ready to be reverted."""

    def ready_checker(succ_connected_it):
        for succ in succ_connected_it:
            succ_atom, (succ_atom_state, _succ_atom_intention) = succ
            if succ_atom_state not in (st.PENDING, st.REVERTED, st.IGNORE):
                LOG.trace("Unable to begin to revert since successor atom '%s' is in state %s", succ_atom, succ_atom_state)
                return False
        LOG.trace("Able to let '%s' revert", atom)
        return True
    noop_decider = deciders.NoOpDecider()
    connected_fetcher = lambda: traversal.depth_first_iterate(self._execution_graph, atom, traversal.Direction.FORWARD)
    decider_fetcher = lambda: noop_decider
    LOG.trace("Checking if '%s' is ready to revert", atom)
    return self._get_maybe_ready(atom, st.REVERTING, [st.REVERT, st.RETRY], connected_fetcher, ready_checker, decider_fetcher, for_what='revert')