import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
class CommentValue(object):

    def __init__(self, val, comment, beginline, _dict):
        self.val = val
        separator = '\n' if beginline else ' '
        self.comment = separator + comment
        self._dict = _dict

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, value):
        self.val[key] = value

    def dump(self, dump_value_func):
        retstr = dump_value_func(self.val)
        if isinstance(self.val, self._dict):
            return self.comment + '\n' + unicode(retstr)
        else:
            return unicode(retstr) + self.comment