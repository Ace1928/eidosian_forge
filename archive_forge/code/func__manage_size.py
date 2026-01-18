import re
from six.moves import html_entities as entities
import six
def _manage_size(self):
    while len(self._dict) > self.capacity:
        del self._dict[self.tail.key]
        if self.tail != self.head:
            self.tail = self.tail.prv
            self.tail.nxt = None
        else:
            self.head = self.tail = None