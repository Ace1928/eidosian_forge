import codecs
import contextlib
import locale
import logging
import math
import os
from functools import partial
from typing import TextIO, Union
import dill
class TraceFormatter(logging.Formatter):
    """
    Generates message prefix and suffix from record.

    This Formatter adds prefix and suffix strings to the log message in trace
    mode (an also provides empty string defaults for normal logs).
    """

    def __init__(self, *args, handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            encoding = handler.stream.encoding
            if encoding is None:
                raise AttributeError
        except AttributeError:
            encoding = locale.getpreferredencoding()
        try:
            encoding = codecs.lookup(encoding).name
        except LookupError:
            self.is_utf8 = False
        else:
            self.is_utf8 = encoding == codecs.lookup('utf-8').name

    def format(self, record):
        fields = {'prefix': '', 'suffix': ''}
        if getattr(record, 'depth', 0) > 0:
            if record.msg.startswith('#'):
                prefix = (record.depth - 1) * '│' + '└'
            elif record.depth == 1:
                prefix = '┬'
            else:
                prefix = (record.depth - 2) * '│' + '├┬'
            if not self.is_utf8:
                prefix = prefix.translate(ASCII_MAP) + '-'
            fields['prefix'] = prefix + ' '
        if hasattr(record, 'size') and record.size is not None and (record.size >= 1):
            power = int(math.log(record.size, 2)) // 10
            size = record.size >> power * 10
            fields['suffix'] = ' [%d %sB]' % (size, 'KMGTP'[power] + 'i' if power else '')
        vars(record).update(fields)
        return super().format(record)