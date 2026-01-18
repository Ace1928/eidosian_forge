from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
class MessageFilters:

    def __init__(self):
        self.filters = {}
        self.filter_ids = count()

    def matches(self, message):
        for handle in self.filters.values():
            if handle.rule.matches(message):
                yield handle