from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _splitPrefix(name):
    """Internal method for splitting a prefixed Element name into its
    respective parts"""
    ntok = name.split(':', 1)
    if len(ntok) == 2:
        return ntok
    else:
        return (None, ntok[0])