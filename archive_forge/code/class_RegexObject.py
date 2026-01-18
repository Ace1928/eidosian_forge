import datetime
import re
from collections import namedtuple
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT
class RegexObject:

    def __init__(self, obj):
        self.pattern = obj.pattern
        self.flags = obj.flags

    def __eq__(self, other):
        if not isinstance(other, RegexObject):
            return NotImplemented
        return self.pattern == other.pattern and self.flags == other.flags