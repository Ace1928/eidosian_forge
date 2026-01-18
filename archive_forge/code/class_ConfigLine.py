from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
class ConfigLine(object):

    def __init__(self, raw):
        self.text = str(raw).strip()
        self.raw = raw
        self._children = list()
        self._parents = list()

    def __str__(self):
        return self.raw

    def __eq__(self, other):
        return self.line == other.line

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, key):
        for item in self._children:
            if item.text == key:
                return item
        raise KeyError(key)

    @property
    def line(self):
        line = self.parents
        line.append(self.text)
        return ' '.join(line)

    @property
    def children(self):
        return _obj_to_text(self._children)

    @property
    def child_objs(self):
        return self._children

    @property
    def parents(self):
        return _obj_to_text(self._parents)

    @property
    def path(self):
        config = _obj_to_raw(self._parents)
        config.append(self.raw)
        return '\n'.join(config)

    @property
    def has_children(self):
        return len(self._children) > 0

    @property
    def has_parents(self):
        return len(self._parents) > 0

    def add_child(self, obj):
        if not isinstance(obj, ConfigLine):
            raise AssertionError('child must be of type `ConfigLine`')
        self._children.append(obj)