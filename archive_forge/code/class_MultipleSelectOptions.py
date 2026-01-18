import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class MultipleSelectOptions(SetMixin):
    """
    Represents all the selected options in a ``<select multiple>`` element.

    You can add to this set-like option to select an option, or remove
    to unselect the option.
    """

    def __init__(self, select):
        self.select = select

    @property
    def options(self):
        """
        Iterator of all the ``<option>`` elements.
        """
        return iter(_options_xpath(self.select))

    def __iter__(self):
        for option in self.options:
            if 'selected' in option.attrib:
                opt_value = option.get('value')
                if opt_value is None:
                    opt_value = (option.text or '').strip()
                yield opt_value

    def add(self, item):
        for option in self.options:
            opt_value = option.get('value')
            if opt_value is None:
                opt_value = (option.text or '').strip()
            if opt_value == item:
                option.set('selected', '')
                break
        else:
            raise ValueError('There is no option with the value %r' % item)

    def remove(self, item):
        for option in self.options:
            opt_value = option.get('value')
            if opt_value is None:
                opt_value = (option.text or '').strip()
            if opt_value == item:
                if 'selected' in option.attrib:
                    del option.attrib['selected']
                else:
                    raise ValueError('The option %r is not currently selected' % item)
                break
        else:
            raise ValueError('There is not option with the value %r' % item)

    def __repr__(self):
        return '<%s {%s} for select name=%r>' % (self.__class__.__name__, ', '.join([repr(v) for v in self]), self.select.name)