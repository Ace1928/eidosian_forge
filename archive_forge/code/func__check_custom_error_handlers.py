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
def _check_custom_error_handlers(self):
    messages = []
    for status_code, num_parameters in [(400, 2), (403, 2), (404, 2), (500, 1)]:
        try:
            handler = self.resolve_error_handler(status_code)
        except (ImportError, ViewDoesNotExist) as e:
            path = getattr(self.urlconf_module, 'handler%s' % status_code)
            msg = "The custom handler{status_code} view '{path}' could not be imported.".format(status_code=status_code, path=path)
            messages.append(Error(msg, hint=str(e), id='urls.E008'))
            continue
        signature = inspect.signature(handler)
        args = [None] * num_parameters
        try:
            signature.bind(*args)
        except TypeError:
            msg = "The custom handler{status_code} view '{path}' does not take the correct number of arguments ({args}).".format(status_code=status_code, path=handler.__module__ + '.' + handler.__qualname__, args='request, exception' if num_parameters == 2 else 'request')
            messages.append(Error(msg, id='urls.E007'))
    return messages