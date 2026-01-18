import time
import types
from ..trace import mutter
from ..transport import decorator
def _show_result(self, before, methodname, result):
    result_len = None
    if isinstance(result, types.GeneratorType):
        result = list(result)
        return_result = iter(result)
    else:
        return_result = result
    getvalue = getattr(result, 'getvalue', None)
    if getvalue is not None:
        val = repr(getvalue())
        result_len = len(val)
        shown_result = '%s(%s) (%d bytes)' % (result.__class__.__name__, self._shorten(val), result_len)
    elif methodname == 'readv':
        num_hunks = len(result)
        total_bytes = sum((len(d) for o, d in result))
        shown_result = 'readv response, %d hunks, %d total bytes' % (num_hunks, total_bytes)
        result_len = total_bytes
    else:
        shown_result = self._shorten(self._strip_tuple_parens(result))
    mutter('  --> %s' % shown_result)
    if False:
        elapsed = time.time() - before
        if result_len and elapsed > 0:
            mutter('      %9.03fs %8dkB/s' % (elapsed, result_len / elapsed / 1000))
        else:
            mutter('      %9.03fs' % elapsed)
    return return_result