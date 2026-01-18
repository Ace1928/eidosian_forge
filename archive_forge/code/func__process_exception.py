from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def _process_exception(request, exception):
    if hasattr(middleware, 'process_exception'):
        result = middleware.process_exception(request, exception)
        if result is not None:
            return result
    raise