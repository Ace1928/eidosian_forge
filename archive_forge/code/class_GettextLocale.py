import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES
from typing import Iterable, Any, Union, Dict, Optional
class GettextLocale(Locale):
    """Locale implementation using the `gettext` module."""

    def __init__(self, code: str, translations: gettext.NullTranslations) -> None:
        self.ngettext = translations.ngettext
        self.gettext = translations.gettext
        super().__init__(code)

    def translate(self, message: str, plural_message: Optional[str]=None, count: Optional[int]=None) -> str:
        if plural_message is not None:
            assert count is not None
            return self.ngettext(message, plural_message, count)
        else:
            return self.gettext(message)

    def pgettext(self, context: str, message: str, plural_message: Optional[str]=None, count: Optional[int]=None) -> str:
        """Allows to set context for translation, accepts plural forms.

        Usage example::

            pgettext("law", "right")
            pgettext("good", "right")

        Plural message example::

            pgettext("organization", "club", "clubs", len(clubs))
            pgettext("stick", "club", "clubs", len(clubs))

        To generate POT file with context, add following options to step 1
        of `load_gettext_translations` sequence::

            xgettext [basic options] --keyword=pgettext:1c,2 --keyword=pgettext:1c,2,3

        .. versionadded:: 4.2
        """
        if plural_message is not None:
            assert count is not None
            msgs_with_ctxt = ('%s%s%s' % (context, CONTEXT_SEPARATOR, message), '%s%s%s' % (context, CONTEXT_SEPARATOR, plural_message), count)
            result = self.ngettext(*msgs_with_ctxt)
            if CONTEXT_SEPARATOR in result:
                result = self.ngettext(message, plural_message, count)
            return result
        else:
            msg_with_ctxt = '%s%s%s' % (context, CONTEXT_SEPARATOR, message)
            result = self.gettext(msg_with_ctxt)
            if CONTEXT_SEPARATOR in result:
                result = message
            return result