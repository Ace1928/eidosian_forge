from __future__ import annotations
import functools
from typing import Any, Callable, Final, TypeVar, cast
import streamlit
from streamlit import config
from streamlit.logger import get_logger
Create a wrapper for an object that has been deprecated. The first
    time one of the object's properties or functions is accessed, the
    given `show_warning` callback will be called.
    