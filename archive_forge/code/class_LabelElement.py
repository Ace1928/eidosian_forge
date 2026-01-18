import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class LabelElement(HtmlElement):
    """
    Represents a ``<label>`` element.

    Label elements are linked to other elements with their ``for``
    attribute.  You can access this element with ``label.for_element``.
    """

    @property
    def for_element(self):
        """
        Get/set the element this label points to.  Return None if it
        can't be found.
        """
        id = self.get('for')
        if not id:
            return None
        return self.body.get_element_by_id(id)

    @for_element.setter
    def for_element(self, other):
        id = other.get('id')
        if not id:
            raise TypeError('Element %r has no id attribute' % other)
        self.set('for', id)

    @for_element.deleter
    def for_element(self):
        attrib = self.attrib
        if 'id' in attrib:
            del attrib['id']