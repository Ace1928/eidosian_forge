from taskflow.engines.action_engine.actions import base
from taskflow import retry as retry_atom
from taskflow import states
from taskflow.types import failure
def _get_retry_args(self, retry, revert=False, addons=None):
    if revert:
        arguments = self._storage.fetch_mapped_args(retry.revert_rebind, atom_name=retry.name, optional_args=retry.revert_optional)
    else:
        arguments = self._storage.fetch_mapped_args(retry.rebind, atom_name=retry.name, optional_args=retry.optional)
    history = self._storage.get_retry_history(retry.name)
    arguments[retry_atom.EXECUTE_REVERT_HISTORY] = history
    if addons:
        arguments.update(addons)
    return arguments