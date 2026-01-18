from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def _update_method_wrapper(_wrapper, decorator):

    @decorator
    def dummy(*args, **kwargs):
        pass
    update_wrapper(_wrapper, dummy)