import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class CheckboxGroup(list):
    """
    Represents a group of checkboxes (``<input type=checkbox>``) that
    have the same name.

    In addition to using this like a list, the ``.value`` attribute
    returns a set-like object that you can add to or remove from to
    check and uncheck checkboxes.  You can also use ``.value_options``
    to get the possible values.
    """

    @property
    def value(self):
        """
        Return a set-like object that can be modified to check or
        uncheck individual checkboxes according to their value.
        """
        return CheckboxValues(self)

    @value.setter
    def value(self, value):
        values = self.value
        values.clear()
        if not hasattr(value, '__iter__'):
            raise ValueError('A CheckboxGroup (name=%r) must be set to a sequence (not %r)' % (self[0].name, value))
        values.update(value)

    @value.deleter
    def value(self):
        self.value.clear()

    @property
    def value_options(self):
        """
        Returns a list of all the possible values.
        """
        return [el.get('value') for el in self]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, list.__repr__(self))