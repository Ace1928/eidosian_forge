import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def _browse_atoms_for_revert(self, atom=None):
    """Browse next atoms to revert.

        This returns a iterator of atoms that *may* be ready to be be
        reverted, if given a specific atom it will only examine the
        predecessors of that atom, otherwise it will examine the whole
        graph.
        """
    if atom is None:
        atom_it = self._runtime.iterate_nodes(co.ATOMS)
    else:
        atom_it = traversal.breadth_first_iterate(self._execution_graph, atom, traversal.Direction.BACKWARD, through_retries=False)
    for atom in atom_it:
        is_ready, late_decider = self._get_maybe_ready_for_revert(atom)
        if is_ready:
            yield (atom, late_decider)