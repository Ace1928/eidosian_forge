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
def _text_words(self, length, truncate):
    """
        Truncate a string after a certain number of words.

        Strip newlines in the string.
        """
    words = self._wrapped.split()
    if len(words) > length:
        words = words[:length]
        return add_truncation_text(' '.join(words), truncate)
    return ' '.join(words)