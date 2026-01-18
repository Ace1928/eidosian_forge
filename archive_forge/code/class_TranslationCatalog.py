import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from asgiref.local import Local
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe
from . import to_language, to_locale
class TranslationCatalog:
    """
    Simulate a dict for DjangoTranslation._catalog so as multiple catalogs
    with different plural equations are kept separate.
    """

    def __init__(self, trans=None):
        self._catalogs = [trans._catalog.copy()] if trans else [{}]
        self._plurals = [trans.plural] if trans else [lambda n: int(n != 1)]

    def __getitem__(self, key):
        for cat in self._catalogs:
            try:
                return cat[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._catalogs[0][key] = value

    def __contains__(self, key):
        return any((key in cat for cat in self._catalogs))

    def items(self):
        for cat in self._catalogs:
            yield from cat.items()

    def keys(self):
        for cat in self._catalogs:
            yield from cat.keys()

    def update(self, trans):
        for cat, plural in zip(self._catalogs, self._plurals):
            if trans.plural.__code__ == plural.__code__:
                cat.update(trans._catalog)
                break
        else:
            self._catalogs.insert(0, trans._catalog.copy())
            self._plurals.insert(0, trans.plural)

    def get(self, key, default=None):
        missing = object()
        for cat in self._catalogs:
            result = cat.get(key, missing)
            if result is not missing:
                return result
        return default

    def plural(self, msgid, num):
        for cat, plural in zip(self._catalogs, self._plurals):
            tmsg = cat.get((msgid, plural(num)))
            if tmsg is not None:
                return tmsg
        raise KeyError