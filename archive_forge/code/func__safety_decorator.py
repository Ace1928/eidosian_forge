from functools import wraps
from django.utils.functional import keep_lazy
def _safety_decorator(safety_marker, func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        return safety_marker(func(*args, **kwargs))
    return wrapper