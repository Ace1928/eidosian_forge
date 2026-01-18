import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def _get_backend_list(self, loaded=False):
    """Return the list of available backends for FlexiBLAS.

        If loaded is False, return the list of available backends from the FlexiBLAS
        configuration. If loaded is True, return the list of actually loaded backends.
        """
    func_name = f'flexiblas_list{('_loaded' if loaded else '')}'
    get_backend_list_ = getattr(self.dynlib, func_name, None)
    if get_backend_list_ is None:
        return None
    n_backends = get_backend_list_(None, 0, 0)
    backends = []
    for i in range(n_backends):
        backend_name = ctypes.create_string_buffer(1024)
        get_backend_list_(backend_name, 1024, i)
        if backend_name.value.decode('utf-8') != '__FALLBACK__':
            backends.append(backend_name.value.decode('utf-8'))
    return backends