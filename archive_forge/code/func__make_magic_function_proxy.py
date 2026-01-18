from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
@staticmethod
def _make_magic_function_proxy(name):

    def proxy(self, *args):
        maybe_show_warning()
        return getattr(obj, name)
    return proxy