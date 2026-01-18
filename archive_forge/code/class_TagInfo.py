from __future__ import annotations
from collections import namedtuple
class TagInfo(namedtuple('_TagInfo', 'value name type length enum')):
    __slots__ = []

    def __new__(cls, value=None, name='unknown', type=None, length=None, enum=None):
        return super().__new__(cls, value, name, type, length, enum or {})

    def cvt_enum(self, value):
        return self.enum.get(value, value) if self.enum else value