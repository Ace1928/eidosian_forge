import time
import types
from ..trace import mutter
from ..transport import decorator
def _call_and_log_result(self, methodname, args, kwargs):
    before = time.time()
    try:
        result = getattr(self._decorated, methodname)(*args, **kwargs)
    except Exception as e:
        mutter('  --> %s' % e)
        mutter('      %.03fs' % (time.time() - before))
        raise
    return self._show_result(before, methodname, result)