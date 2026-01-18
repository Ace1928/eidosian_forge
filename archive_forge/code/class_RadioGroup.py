import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class RadioGroup(list):
    """
    This object represents several ``<input type=radio>`` elements
    that have the same name.

    You can use this like a list, but also use the property
    ``.value`` to check/uncheck inputs.  Also you can use
    ``.value_options`` to get the possible values.
    """

    @property
    def value(self):
        """
        Get/set the value, which checks the radio with that value (and
        unchecks any other value).
        """
        for el in self:
            if 'checked' in el.attrib:
                return el.get('value')
        return None

    @value.setter
    def value(self, value):
        checked_option = None
        if value is not None:
            for el in self:
                if el.get('value') == value:
                    checked_option = el
                    break
            else:
                raise ValueError('There is no radio input with the value %r' % value)
        for el in self:
            if 'checked' in el.attrib:
                del el.attrib['checked']
        if checked_option is not None:
            checked_option.set('checked', '')

    @value.deleter
    def value(self):
        self.value = None

    @property
    def value_options(self):
        """
        Returns a list of all the possible values.
        """
        return [el.get('value') for el in self]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, list.__repr__(self))