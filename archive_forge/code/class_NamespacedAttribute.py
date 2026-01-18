import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class NamespacedAttribute(str):
    """A namespaced string (e.g. 'xml:lang') that remembers the namespace
    ('xml') and the name ('lang') that were used to create it.
    """

    def __new__(cls, prefix, name=None, namespace=None):
        if not name:
            name = None
        if not name:
            obj = str.__new__(cls, prefix)
        elif not prefix:
            obj = str.__new__(cls, name)
        else:
            obj = str.__new__(cls, prefix + ':' + name)
        obj.prefix = prefix
        obj.name = name
        obj.namespace = namespace
        return obj