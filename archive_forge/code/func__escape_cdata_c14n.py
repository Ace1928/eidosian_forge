import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _escape_cdata_c14n(text):
    try:
        if '&' in text:
            text = text.replace('&', '&amp;')
        if '<' in text:
            text = text.replace('<', '&lt;')
        if '>' in text:
            text = text.replace('>', '&gt;')
        if '\r' in text:
            text = text.replace('\r', '&#xD;')
        return text
    except (TypeError, AttributeError):
        _raise_serialization_error(text)