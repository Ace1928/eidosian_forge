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
class DjangoTranslation(gettext_module.GNUTranslations):
    """
    Set up the GNUTranslations context with regard to output charset.

    This translation object will be constructed out of multiple GNUTranslations
    objects by merging their catalogs. It will construct an object for the
    requested language and add a fallback to the default language, if it's
    different from the requested language.
    """
    domain = 'django'

    def __init__(self, language, domain=None, localedirs=None):
        """Create a GNUTranslations() using many locale directories"""
        gettext_module.GNUTranslations.__init__(self)
        if domain is not None:
            self.domain = domain
        self.__language = language
        self.__to_language = to_language(language)
        self.__locale = to_locale(language)
        self._catalog = None
        self.plural = lambda n: int(n != 1)
        if self.domain == 'django':
            if localedirs is not None:
                warnings.warn("localedirs is ignored when domain is 'django'.", RuntimeWarning)
                localedirs = None
            self._init_translation_catalog()
        if localedirs:
            for localedir in localedirs:
                translation = self._new_gnu_trans(localedir)
                self.merge(translation)
        else:
            self._add_installed_apps_translations()
        self._add_local_translations()
        if self.__language == settings.LANGUAGE_CODE and self.domain == 'django' and (self._catalog is None):
            raise OSError('No translation files found for default language %s.' % settings.LANGUAGE_CODE)
        self._add_fallback(localedirs)
        if self._catalog is None:
            self._catalog = TranslationCatalog()

    def __repr__(self):
        return '<DjangoTranslation lang:%s>' % self.__language

    def _new_gnu_trans(self, localedir, use_null_fallback=True):
        """
        Return a mergeable gettext.GNUTranslations instance.

        A convenience wrapper. By default gettext uses 'fallback=False'.
        Using param `use_null_fallback` to avoid confusion with any other
        references to 'fallback'.
        """
        return gettext_module.translation(domain=self.domain, localedir=localedir, languages=[self.__locale], fallback=use_null_fallback)

    def _init_translation_catalog(self):
        """Create a base catalog using global django translations."""
        settingsfile = sys.modules[settings.__module__].__file__
        localedir = os.path.join(os.path.dirname(settingsfile), 'locale')
        translation = self._new_gnu_trans(localedir)
        self.merge(translation)

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

    def _add_local_translations(self):
        """Merge translations defined in LOCALE_PATHS."""
        for localedir in reversed(settings.LOCALE_PATHS):
            translation = self._new_gnu_trans(localedir)
            self.merge(translation)

    def _add_fallback(self, localedirs=None):
        """Set the GNUTranslations() fallback with the default language."""
        if self.__language == settings.LANGUAGE_CODE or self.__language.startswith('en'):
            return
        if self.domain == 'django':
            default_translation = translation(settings.LANGUAGE_CODE)
        else:
            default_translation = DjangoTranslation(settings.LANGUAGE_CODE, domain=self.domain, localedirs=localedirs)
        self.add_fallback(default_translation)

    def merge(self, other):
        """Merge another translation into this catalog."""
        if not getattr(other, '_catalog', None):
            return
        if self._catalog is None:
            self.plural = other.plural
            self._info = other._info.copy()
            self._catalog = TranslationCatalog(other)
        else:
            self._catalog.update(other)
        if other._fallback:
            self.add_fallback(other._fallback)

    def language(self):
        """Return the translation language."""
        return self.__language

    def to_language(self):
        """Return the translation language name."""
        return self.__to_language

    def ngettext(self, msgid1, msgid2, n):
        try:
            tmsg = self._catalog.plural(msgid1, n)
        except KeyError:
            if self._fallback:
                return self._fallback.ngettext(msgid1, msgid2, n)
            if n == 1:
                tmsg = msgid1
            else:
                tmsg = msgid2
        return tmsg