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
class FlexiBLASController(LibController):
    """Controller class for FlexiBLAS"""
    user_api = 'blas'
    internal_api = 'flexiblas'
    filename_prefixes = ('libflexiblas',)
    check_symbols = ('flexiblas_get_num_threads', 'flexiblas_set_num_threads', 'flexiblas_get_version', 'flexiblas_list', 'flexiblas_list_loaded', 'flexiblas_current_backend')

    @property
    def loaded_backends(self):
        return self._get_backend_list(loaded=True)

    @property
    def current_backend(self):
        return self._get_current_backend()

    def info(self):
        """Return relevant info wrapped in a dict"""
        exposed_attrs = super().info()
        exposed_attrs['loaded_backends'] = self.loaded_backends
        exposed_attrs['current_backend'] = self.current_backend
        return exposed_attrs

    def set_additional_attributes(self):
        self.available_backends = self._get_backend_list(loaded=False)

    def get_num_threads(self):
        get_func = getattr(self.dynlib, 'flexiblas_get_num_threads', lambda: None)
        num_threads = get_func()
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, 'flexiblas_set_num_threads', lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        get_version_ = getattr(self.dynlib, 'flexiblas_get_version', None)
        if get_version_ is None:
            return None
        major = ctypes.c_int()
        minor = ctypes.c_int()
        patch = ctypes.c_int()
        get_version_(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        return f'{major.value}.{minor.value}.{patch.value}'

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

    def _get_current_backend(self):
        """Return the backend of FlexiBLAS"""
        get_backend_ = getattr(self.dynlib, 'flexiblas_current_backend', None)
        if get_backend_ is None:
            return None
        backend = ctypes.create_string_buffer(1024)
        get_backend_(backend, ctypes.sizeof(backend))
        return backend.value.decode('utf-8')

    def switch_backend(self, backend):
        """Switch the backend of FlexiBLAS

        Parameters
        ----------
        backend : str
            The name or the path to the shared library of the backend to switch to. If
            the backend is not already loaded, it will be loaded first.
        """
        if backend not in self.loaded_backends:
            if backend in self.available_backends:
                load_func = getattr(self.dynlib, 'flexiblas_load_backend', lambda _: -1)
            else:
                load_func = getattr(self.dynlib, 'flexiblas_load_backend_library', lambda _: -1)
            res = load_func(str(backend).encode('utf-8'))
            if res == -1:
                raise RuntimeError(f'Failed to load backend {backend!r}. It must either be the name of a backend available in the FlexiBLAS configuration {self.available_backends} or the path to a valid shared library.')
            self.parent._load_libraries()
        switch_func = getattr(self.dynlib, 'flexiblas_switch', lambda _: -1)
        idx = self.loaded_backends.index(backend)
        res = switch_func(idx)
        if res == -1:
            raise RuntimeError(f'Failed to switch to backend {backend!r}.')