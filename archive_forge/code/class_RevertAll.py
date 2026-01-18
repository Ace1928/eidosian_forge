import abc
import weakref
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import executor as ex
from taskflow import logging
from taskflow import retry as retry_atom
from taskflow import states as st
class RevertAll(Strategy):
    """Sets *all* nodes/atoms to the ``REVERT`` intention."""
    strategy = retry_atom.REVERT_ALL

    def __init__(self, runtime):
        super(RevertAll, self).__init__(runtime)

    def apply(self):
        return self._runtime.reset_atoms(self._runtime.iterate_nodes(co.ATOMS), state=None, intention=st.REVERT)