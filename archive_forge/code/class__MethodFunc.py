import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class _MethodFunc:
    """
    An object that represents a method on an element as a function;
    the function takes either an element or an HTML string.  It
    returns whatever the function normally returns, or if the function
    works in-place (and so returns None) it returns a serialized form
    of the resulting document.
    """

    def __init__(self, name, copy=False, source_class=HtmlMixin):
        self.name = name
        self.copy = copy
        self.__doc__ = getattr(source_class, self.name).__doc__

    def __call__(self, doc, *args, **kw):
        result_type = type(doc)
        if isinstance(doc, str):
            if 'copy' in kw:
                raise TypeError("The keyword 'copy' can only be used with element inputs to %s, not a string input" % self.name)
            doc = fromstring(doc, **kw)
        else:
            if 'copy' in kw:
                make_a_copy = kw.pop('copy')
            else:
                make_a_copy = self.copy
            if make_a_copy:
                doc = copy.deepcopy(doc)
        meth = getattr(doc, self.name)
        result = meth(*args, **kw)
        if result is None:
            return _transform_result(result_type, doc)
        else:
            return result