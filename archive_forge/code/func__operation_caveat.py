import collections
import pyrfc3339
from ._conditions import (
def _operation_caveat(cond, ops):
    """ Helper for allow_caveat and deny_caveat.

    It checks that all operation names are valid before creating the caveat.
    """
    for op in ops:
        if op.find(' ') != -1:
            return error_caveat('invalid operation name "{}"'.format(op))
    return _first_party(cond, ' '.join(ops))