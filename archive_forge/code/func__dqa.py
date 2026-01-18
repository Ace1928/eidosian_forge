from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _dqa(self, attr):
    """Dequalify an attribute key as needed"""
    if isinstance(attr, tuple) and (not attr[0]):
        return attr[1]
    else:
        return attr