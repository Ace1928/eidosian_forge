import sys
import os
import re
import warnings
import types
import unicodedata
def _dom_node(self, domroot):
    element = domroot.createElement(self.tagname)
    for attribute, value in self.attlist():
        if isinstance(value, list):
            value = ' '.join([serial_escape('%s' % (v,)) for v in value])
        element.setAttribute(attribute, '%s' % value)
    for child in self.children:
        element.appendChild(child._dom_node(domroot))
    return element