from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
@classmethod
def _iter_patch_candidates(cls):
    """helper to scan for monkeypatches.

        returns tuple containing:
        * object (module or class)
        * attribute of object
        * value of attribute
        * whether it should or should not be patched
        """
    from django.contrib.auth import models, hashers
    user_attrs = ['check_password', 'set_password']
    model_attrs = ['check_password', 'make_password']
    hasher_attrs = ['check_password', 'make_password', 'get_hasher', 'identify_hasher', 'get_hashers']
    objs = [(models, model_attrs), (models.User, user_attrs), (hashers, hasher_attrs)]
    for obj, patched in objs:
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            value = obj.__dict__.get(attr, UNSET)
            if value is UNSET and attr not in patched:
                continue
            value = get_method_function(value)
            source = getattr(value, '__module__', None)
            if source:
                yield (obj, attr, source, attr in patched)