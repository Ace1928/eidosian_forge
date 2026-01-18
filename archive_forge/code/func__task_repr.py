import linecache
import reprlib
import traceback
from . import base_futures
from . import coroutines
@reprlib.recursive_repr()
def _task_repr(task):
    info = ' '.join(_task_repr_info(task))
    return f'<{task.__class__.__name__} {info}>'