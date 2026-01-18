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
def _add_installed_apps_translations(self):
    """Merge translations from each installed app."""
    try:
        app_configs = reversed(apps.get_app_configs())
    except AppRegistryNotReady:
        raise AppRegistryNotReady("The translation infrastructure cannot be initialized before the apps registry is ready. Check that you don't make non-lazy gettext calls at import time.")
    for app_config in app_configs:
        localedir = os.path.join(app_config.path, 'locale')
        if os.path.exists(localedir):
            translation = self._new_gnu_trans(localedir)
            self.merge(translation)