import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
def _get_next_value(self, values, history):
    remaining = misc.sequence_minus(values, history.provided_iter())
    if not remaining:
        raise exc.NotFound('No elements left in collection of iterable retry controller %s' % self.name)
    return remaining[0]