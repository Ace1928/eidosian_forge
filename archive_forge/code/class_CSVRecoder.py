from __future__ import (absolute_import, division, print_function)
import codecs
import csv
from collections.abc import MutableSequence
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.six import PY2
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
class CSVRecoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """

    def __init__(self, f, encoding='utf-8'):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.reader).encode('utf-8')
    next = __next__