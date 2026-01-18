import re
import sys
from html import escape
from urllib.parse import quote
from paste.util.looper import looper
class html(object):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.value)