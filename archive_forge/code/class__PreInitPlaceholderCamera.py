import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pygame import error
class _PreInitPlaceholderCamera(AbstractCamera):
    __init__ = _pre_init_placeholder_varargs
    start = _pre_init_placeholder_varargs
    stop = _pre_init_placeholder_varargs
    get_controls = _pre_init_placeholder_varargs
    set_controls = _pre_init_placeholder_varargs
    get_size = _pre_init_placeholder_varargs
    query_image = _pre_init_placeholder_varargs
    get_image = _pre_init_placeholder_varargs
    get_raw = _pre_init_placeholder_varargs