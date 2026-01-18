import re
from six.moves import html_entities as entities
import six
def _insert_item(self, item):
    item.prv = None
    item.nxt = self.head
    if self.head is not None:
        self.head.prv = item
    else:
        self.tail = item
    self.head = item
    self._manage_size()