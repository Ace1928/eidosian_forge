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
class translator:
    """
    Translations manager.
    """
    _TRANSLATORS: dict[str, TranslationBundle] = {}
    _LOCALE = SYS_LOCALE

    @staticmethod
    def normalize_domain(domain: str) -> str:
        """Normalize a domain name.

        Parameters
        ----------
        domain: str
            Domain to normalize

        Returns
        -------
        str
            Normalized domain
        """
        return domain.replace('-', '_')

    @classmethod
    def set_locale(cls, locale_: str) -> None:
        """
        Set locale for the translation bundles based on the settings.

        Parameters
        ----------
        locale_: str
            The language name to use.
        """
        if locale_ == cls._LOCALE:
            return
        if is_valid_locale(locale_):
            cls._LOCALE = locale_
            for _, bundle in cls._TRANSLATORS.items():
                bundle.update_locale(locale_)

    @classmethod
    def load(cls, domain: str) -> TranslationBundle:
        """
        Load translation domain.

        The domain is usually the normalized ``package_name``.

        Parameters
        ----------
        domain: str
            The translations domain. The normalized python package name.

        Returns
        -------
        Translator
            A translator instance bound to the domain.
        """
        norm_domain = translator.normalize_domain(domain)
        if norm_domain in cls._TRANSLATORS:
            trans = cls._TRANSLATORS[norm_domain]
        else:
            trans = TranslationBundle(norm_domain, cls._LOCALE)
            cls._TRANSLATORS[norm_domain] = trans
        return trans

    @staticmethod
    def _translate_schema_strings(translations: Any, schema: dict, prefix: str='', to_translate: dict[Pattern, str] | None=None) -> None:
        """Translate a schema in-place."""
        if to_translate is None:
            to_translate = _prepare_schema_patterns(schema)
        for key, value in schema.items():
            path = prefix + '/' + key
            if isinstance(value, str):
                matched = False
                for pattern, context in to_translate.items():
                    if pattern.fullmatch(path):
                        matched = True
                        break
                if matched:
                    schema[key] = translations.pgettext(context, value)
            elif isinstance(value, dict):
                translator._translate_schema_strings(translations, value, prefix=path, to_translate=to_translate)
            elif isinstance(value, list):
                for i, element in enumerate(value):
                    if not isinstance(element, dict):
                        continue
                    translator._translate_schema_strings(translations, element, prefix=path + '[' + str(i) + ']', to_translate=to_translate)

    @staticmethod
    def translate_schema(schema: dict) -> dict:
        """Translate a schema.

        Parameters
        ----------
        schema: dict
            The schema to be translated

        Returns
        -------
        Dict
            The translated schema
        """
        if translator._LOCALE == DEFAULT_LOCALE:
            return schema
        translations = translator.load(schema.get(_lab_i18n_config, {}).get('domain', DEFAULT_DOMAIN))
        new_schema = schema.copy()
        translator._translate_schema_strings(translations, new_schema)
        return new_schema