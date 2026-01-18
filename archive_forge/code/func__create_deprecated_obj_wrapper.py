from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
def _create_deprecated_obj_wrapper(obj: TObj, show_warning: Callable[[], Any]) -> TObj:
    """Create a wrapper for an object that has been deprecated. The first
    time one of the object's properties or functions is accessed, the
    given `show_warning` callback will be called.
    """
    has_shown_warning = False

    def maybe_show_warning() -> None:
        nonlocal has_shown_warning
        if not has_shown_warning:
            has_shown_warning = True
            show_warning()

    class Wrapper:

        def __init__(self):
            for name in Wrapper._get_magic_functions(obj.__class__):
                setattr(self.__class__, name, property(self._make_magic_function_proxy(name)))

        def __getattr__(self, attr):
            if attr in self.__dict__:
                return getattr(self, attr)
            maybe_show_warning()
            return getattr(obj, attr)

        @staticmethod
        def _get_magic_functions(cls) -> list[str]:
            ignore = ('__class__', '__dict__', '__getattribute__', '__getattr__')
            return [name for name in dir(cls) if name not in ignore and name.startswith('__')]

        @staticmethod
        def _make_magic_function_proxy(name):

            def proxy(self, *args):
                maybe_show_warning()
                return getattr(obj, name)
            return proxy
    return cast(TObj, Wrapper())