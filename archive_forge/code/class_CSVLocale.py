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
class CSVLocale(Locale):
    """Locale implementation using tornado's CSV translation format."""

    def __init__(self, code: str, translations: Dict[str, Dict[str, str]]) -> None:
        self.translations = translations
        super().__init__(code)

    def translate(self, message: str, plural_message: Optional[str]=None, count: Optional[int]=None) -> str:
        if plural_message is not None:
            assert count is not None
            if count != 1:
                message = plural_message
                message_dict = self.translations.get('plural', {})
            else:
                message_dict = self.translations.get('singular', {})
        else:
            message_dict = self.translations.get('unknown', {})
        return message_dict.get(message, message)

    def pgettext(self, context: str, message: str, plural_message: Optional[str]=None, count: Optional[int]=None) -> str:
        if self.translations:
            gen_log.warning('pgettext is not supported by CSVLocale')
        return self.translate(message, plural_message, count)