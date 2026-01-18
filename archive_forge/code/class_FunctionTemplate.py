from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class FunctionTemplate(ABC):
    unsafe_casting = True
    exact_match_required = False
    prefer_literal = False
    metadata = {}

    def __init__(self, context):
        self.context = context

    def _select(self, cases, args, kws):
        options = {'unsafe_casting': self.unsafe_casting, 'exact_match_required': self.exact_match_required}
        selected = self.context.resolve_overload(self.key, cases, args, kws, **options)
        return selected

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        key = type(self).key
        if isinstance(key, MethodType):
            assert key.im_self is None
            key = key.im_func
        return key

    @classmethod
    def get_source_code_info(cls, impl):
        """
        Gets the source information about function impl.
        Returns:

        code - str: source code as a string
        firstlineno - int: the first line number of the function impl
        path - str: the path to file containing impl

        if any of the above are not available something generic is returned
        """
        try:
            code, firstlineno = inspect.getsourcelines(impl)
        except OSError:
            code = 'None available (built from string?)'
            firstlineno = 0
        path = inspect.getsourcefile(impl)
        if path is None:
            path = '<unknown> (built from string?)'
        return (code, firstlineno, path)

    @abstractmethod
    def get_template_info(self):
        """
        Returns a dictionary with information specific to the template that will
        govern how error messages are displayed to users. The dictionary must
        be of the form:
        info = {
            'kind': "unknown", # str: The kind of template, e.g. "Overload"
            'name': "unknown", # str: The name of the source function
            'sig': "unknown",  # str: The signature(s) of the source function
            'filename': "unknown", # str: The filename of the source function
            'lines': ("start", "end"), # tuple(int, int): The start and
                                         end line of the source function.
            'docstring': "unknown" # str: The docstring of the source function
        }
        """
        pass

    def __str__(self):
        info = self.get_template_info()
        srcinfo = f'{info['filename']}:{info['lines'][0]}'
        return f'<{self.__class__.__name__} {srcinfo}>'
    __repr__ = __str__