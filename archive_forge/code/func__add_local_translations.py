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
def _add_local_translations(self):
    """Merge translations defined in LOCALE_PATHS."""
    for localedir in reversed(settings.LOCALE_PATHS):
        translation = self._new_gnu_trans(localedir)
        self.merge(translation)