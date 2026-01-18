import sys
import os
import re
import warnings
import types
import unicodedata
class system_message(Special, BackLinkable, PreBibliographic, Element):
    """
    System message element.

    Do not instantiate this class directly; use
    ``document.reporter.info/warning/error/severe()`` instead.
    """

    def __init__(self, message=None, *children, **attributes):
        rawsource = attributes.get('rawsource', '')
        if message:
            p = paragraph('', message)
            children = (p,) + children
        try:
            Element.__init__(self, rawsource, *children, **attributes)
        except:
            print('system_message: children=%r' % (children,))
            raise

    def astext(self):
        line = self.get('line', '')
        return '%s:%s: (%s/%s) %s' % (self['source'], line, self['type'], self['level'], Element.astext(self))