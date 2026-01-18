from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
@classmethod
def _get_non_printable_regex(cls, data):
    match = cls.NON_PRINTABLE.search(data)
    if not bool(match):
        return None
    return (match.start(), match.group())