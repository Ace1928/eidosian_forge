import re
from six.moves import html_entities as entities
import six
class _Item(object):

    def __init__(self, key, value):
        self.prv = self.nxt = None
        self.key = key
        self.value = value

    def __repr__(self):
        return repr(self.value)