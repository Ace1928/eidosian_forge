import abc
import weakref
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import executor as ex
from taskflow import logging
from taskflow import retry as retry_atom
from taskflow import states as st
def _determine_resolution(self, atom, failure):
    """Determines which resolution strategy to activate/apply."""
    retry = self._runtime.find_retry(atom)
    if retry is not None:
        handler = self._runtime.fetch_action(retry)
        strategy = handler.on_failure(retry, atom, failure)
        if strategy == retry_atom.RETRY:
            return RevertAndRetry(self._runtime, retry)
        elif strategy == retry_atom.REVERT:
            parent_resolver = self._determine_resolution(retry, failure)
            if self._defer_reverts:
                return parent_resolver
            if parent_resolver is not self._undefined_resolver:
                if parent_resolver.strategy != retry_atom.REVERT:
                    return parent_resolver
            return Revert(self._runtime, retry)
        elif strategy == retry_atom.REVERT_ALL:
            return RevertAll(self._runtime)
        else:
            raise ValueError("Unknown atom failure resolution action/strategy '%s'" % strategy)
    else:
        return self._undefined_resolver