import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class CheckboxValues(SetMixin):
    """
    Represents the values of the checked checkboxes in a group of
    checkboxes with the same name.
    """

    def __init__(self, group):
        self.group = group

    def __iter__(self):
        return iter([el.get('value') for el in self.group if 'checked' in el.attrib])

    def add(self, value):
        for el in self.group:
            if el.get('value') == value:
                el.set('checked', '')
                break
        else:
            raise KeyError('No checkbox with value %r' % value)

    def remove(self, value):
        for el in self.group:
            if el.get('value') == value:
                if 'checked' in el.attrib:
                    del el.attrib['checked']
                else:
                    raise KeyError('The checkbox with value %r was already unchecked' % value)
                break
        else:
            raise KeyError('No checkbox with value %r' % value)

    def __repr__(self):
        return '<%s {%s} for checkboxes name=%r>' % (self.__class__.__name__, ', '.join([repr(v) for v in self]), self.group.name)