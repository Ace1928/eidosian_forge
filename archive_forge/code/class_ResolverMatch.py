import functools
import inspect
import re
import string
from importlib import import_module
from pickle import PicklingError
from urllib.parse import quote
from asgiref.local import Local
from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language
from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
class ResolverMatch:

    def __init__(self, func, args, kwargs, url_name=None, app_names=None, namespaces=None, route=None, tried=None, captured_kwargs=None, extra_kwargs=None):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.url_name = url_name
        self.route = route
        self.tried = tried
        self.captured_kwargs = captured_kwargs
        self.extra_kwargs = extra_kwargs
        self.app_names = [x for x in app_names if x] if app_names else []
        self.app_name = ':'.join(self.app_names)
        self.namespaces = [x for x in namespaces if x] if namespaces else []
        self.namespace = ':'.join(self.namespaces)
        if hasattr(func, 'view_class'):
            func = func.view_class
        if not hasattr(func, '__name__'):
            self._func_path = func.__class__.__module__ + '.' + func.__class__.__name__
        else:
            self._func_path = func.__module__ + '.' + func.__name__
        view_path = url_name or self._func_path
        self.view_name = ':'.join(self.namespaces + [view_path])

    def __getitem__(self, index):
        return (self.func, self.args, self.kwargs)[index]

    def __repr__(self):
        if isinstance(self.func, functools.partial):
            func = repr(self.func)
        else:
            func = self._func_path
        return 'ResolverMatch(func=%s, args=%r, kwargs=%r, url_name=%r, app_names=%r, namespaces=%r, route=%r%s%s)' % (func, self.args, self.kwargs, self.url_name, self.app_names, self.namespaces, self.route, f', captured_kwargs={self.captured_kwargs!r}' if self.captured_kwargs else '', f', extra_kwargs={self.extra_kwargs!r}' if self.extra_kwargs else '')

    def __reduce_ex__(self, protocol):
        raise PicklingError(f'Cannot pickle {self.__class__.__qualname__}.')