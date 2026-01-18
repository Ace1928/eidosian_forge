from functools import wraps
from jedi import debug
class CachedMetaClass(type):
    """
    This is basically almost the same than the decorator above, it just caches
    class initializations. Either you do it this way or with decorators, but
    with decorators you lose class access (isinstance, etc).
    """

    @inference_state_as_method_param_cache()
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)