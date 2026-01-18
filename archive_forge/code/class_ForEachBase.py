import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
class ForEachBase(Retry):
    """Base class for retries that iterate over a given collection."""

    def __init__(self, name=None, provides=None, requires=None, auto_extract=True, rebind=None, revert_all=False):
        super(ForEachBase, self).__init__(name, provides, requires, auto_extract, rebind)
        if revert_all:
            self._revert_action = REVERT_ALL
        else:
            self._revert_action = REVERT

    def _get_next_value(self, values, history):
        remaining = misc.sequence_minus(values, history.provided_iter())
        if not remaining:
            raise exc.NotFound('No elements left in collection of iterable retry controller %s' % self.name)
        return remaining[0]

    def _on_failure(self, values, history):
        try:
            self._get_next_value(values, history)
        except exc.NotFound:
            return self._revert_action
        else:
            return RETRY