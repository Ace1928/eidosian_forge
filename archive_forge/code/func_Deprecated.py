import inspect
import logging
from typing import Optional, Union
from ray.util import log_once
from ray.util.annotations import _mark_annotated
def Deprecated(old=None, *, new=None, help=None, error):
    """Decorator for documenting a deprecated class, method, or function.

    Automatically adds a `deprecation.deprecation_warning(old=...,
    error=False)` to not break existing code at this point to the decorated
    class' constructor, method, or function.

    In a next major release, this warning should then be made an error
    (by setting error=True), which means at this point that the
    class/method/function is no longer supported, but will still inform
    the user about the deprecation event.

    In a further major release, the class, method, function should be erased
    entirely from the codebase.


    .. testcode::
        :skipif: True

        from ray.rllib.utils.deprecation import Deprecated
        # Deprecated class: Patches the constructor to warn if the class is
        # used.
        @Deprecated(new="NewAndMuchCoolerClass", error=False)
        class OldAndUncoolClass:
            ...

        # Deprecated class method: Patches the method to warn if called.
        class StillCoolClass:
            ...
            @Deprecated(new="StillCoolClass.new_and_much_cooler_method()",
                        error=False)
            def old_and_uncool_method(self, uncool_arg):
                ...

        # Deprecated function: Patches the function to warn if called.
        @Deprecated(new="new_and_much_cooler_function", error=False)
        def old_and_uncool_function(*uncool_args):
            ...
    """

    def _inner(obj):
        if inspect.isclass(obj):
            obj_init = obj.__init__

            def patched_init(*args, **kwargs):
                if log_once(old or obj.__name__):
                    deprecation_warning(old=old or obj.__name__, new=new, help=help, error=error)
                return obj_init(*args, **kwargs)
            obj.__init__ = patched_init
            _mark_annotated(obj)
            return obj

        def _ctor(*args, **kwargs):
            if log_once(old or obj.__name__):
                deprecation_warning(old=old or obj.__name__, new=new, help=help, error=error)
            return obj(*args, **kwargs)
        return _ctor
    return _inner