import gzip
import re
import secrets
import unicodedata
from gzip import GzipFile
from gzip import compress as gzip_compress
from io import BytesIO
from django.core.exceptions import SuspiciousFileOperation
from django.utils.functional import SimpleLazyObject, keep_lazy_text, lazy
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy, pgettext
class FakeMatch:
    __slots__ = ['_text', '_end']

    def end(self, group=0):
        assert group == 0, 'This specific object takes only group=0'
        return self._end

    def __getitem__(self, group):
        if group == 1:
            return None
        assert group == 0, 'This specific object takes only group in {0,1}'
        return self._text

    def __init__(self, text, end):
        self._text, self._end = (text, end)