from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
class TranslationBundle:
    """
    Translation bundle providing gettext translation functionality.
    """

    def __init__(self, domain: str, locale_: str):
        """Initialize the bundle."""
        self._domain = domain
        self._locale = locale_
        self._translator = gettext.NullTranslations()
        self.update_locale(locale_)

    def update_locale(self, locale_: str) -> None:
        """
        Update the locale.

        Parameters
        ----------
        locale_: str
            The language name to use.
        """
        self._locale = locale_
        localedir = None
        if locale_ != DEFAULT_LOCALE:
            language_pack_module = f'jupyterlab_language_pack_{locale_}'
            try:
                mod = importlib.import_module(language_pack_module)
                assert mod.__file__ is not None
                localedir = os.path.join(os.path.dirname(mod.__file__), LOCALE_DIR)
            except Exception:
                pass
        self._translator = gettext.translation(self._domain, localedir=localedir, languages=(self._locale,), fallback=True)

    def gettext(self, msgid: str) -> str:
        """
        Translate a singular string.

        Parameters
        ----------
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return self._translator.gettext(msgid)

    def ngettext(self, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Translate a singular string with pluralization.

        Parameters
        ----------
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return self._translator.ngettext(msgid, msgid_plural, n)

    def pgettext(self, msgctxt: str, msgid: str) -> str:
        """
        Translate a singular string with context.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        if PY37_OR_LOWER:
            translation = self._translator.gettext(msgid)
        else:
            translation = self._translator.pgettext(msgctxt, msgid)
        return translation

    def npgettext(self, msgctxt: str, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Translate a singular string with context and pluralization.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        if PY37_OR_LOWER:
            translation = self._translator.ngettext(msgid, msgid_plural, n)
        else:
            translation = self._translator.npgettext(msgctxt, msgid, msgid_plural, n)
        return translation

    def __(self, msgid: str) -> str:
        """
        Shorthand for gettext.

        Parameters
        ----------
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return self.gettext(msgid)

    def _n(self, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Shorthand for ngettext.

        Parameters
        ----------
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return self.ngettext(msgid, msgid_plural, n)

    def _p(self, msgctxt: str, msgid: str) -> str:
        """
        Shorthand for pgettext.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return self.pgettext(msgctxt, msgid)

    def _np(self, msgctxt: str, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Shorthand for npgettext.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return self.npgettext(msgctxt, msgid, msgid_plural, n)