import abc
import weakref
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import executor as ex
from taskflow import logging
from taskflow import retry as retry_atom
from taskflow import states as st
def _process_atom_failure(self, atom, failure):
    """Processes atom failure & applies resolution strategies.

        On atom failure this will find the atoms associated retry controller
        and ask that controller for the strategy to perform to resolve that
        failure. After getting a resolution strategy decision this method will
        then adjust the needed other atoms intentions, and states, ... so that
        the failure can be worked around.
        """
    resolver = self._determine_resolution(atom, failure)
    LOG.debug("Applying resolver '%s' to resolve failure '%s' of atom '%s'", resolver, failure, atom)
    tweaked = resolver.apply()
    if LOG.isEnabledFor(logging.TRACE):
        LOG.trace("Modified/tweaked %s nodes while applying resolver '%s'", tweaked, resolver)
    else:
        LOG.debug("Modified/tweaked %s nodes while applying resolver '%s'", len(tweaked), resolver)