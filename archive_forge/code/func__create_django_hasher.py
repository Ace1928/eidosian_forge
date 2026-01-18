from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def _create_django_hasher(self, django_name):
    """
        helper to create new django hasher by name.
        wraps underlying django methods.
        """
    module = sys.modules.get('passlib.ext.django.models')
    if module is None or not module.adapter.patched:
        from django.contrib.auth.hashers import get_hasher
        try:
            return get_hasher(django_name)
        except ValueError as err:
            if not str(err).startswith('Unknown password hashing algorithm'):
                raise
    else:
        get_hashers = module.adapter._manager.getorig('django.contrib.auth.hashers:get_hashers').__wrapped__
        for hasher in get_hashers():
            if hasher.algorithm == django_name:
                return hasher
    path = self._builtin_django_hashers.get(django_name)
    if path:
        if '.' not in path:
            path = 'django.contrib.auth.hashers.' + path
        from django.utils.module_loading import import_string
        return import_string(path)()
    raise ValueError('unknown hasher: %r' % django_name)