import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def __addLocale(self, key, group=None):
    """add locale to key according the current lc_messages"""
    if not group:
        group = self.defaultGroup
    for lang in xdg.Locale.langs:
        langkey = '%s[%s]' % (key, lang)
        if langkey in self.content[group]:
            return langkey
    return key