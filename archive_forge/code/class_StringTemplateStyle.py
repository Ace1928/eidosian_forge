import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
class StringTemplateStyle(PercentStyle):
    default_format = '${message}'
    asctime_format = '${asctime}'
    asctime_search = '${asctime}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tpl = Template(self._fmt)

    def usesTime(self):
        fmt = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_search) >= 0

    def validate(self):
        pattern = Template.pattern
        fields = set()
        for m in pattern.finditer(self._fmt):
            d = m.groupdict()
            if d['named']:
                fields.add(d['named'])
            elif d['braced']:
                fields.add(d['braced'])
            elif m.group(0) == '$':
                raise ValueError("invalid format: bare '$' not allowed")
        if not fields:
            raise ValueError('invalid format: no fields')

    def _format(self, record):
        if (defaults := self._defaults):
            values = defaults | record.__dict__
        else:
            values = record.__dict__
        return self._tpl.substitute(**values)