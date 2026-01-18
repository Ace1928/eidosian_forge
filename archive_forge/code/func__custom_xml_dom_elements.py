from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
def _custom_xml_dom_elements(self, doc):
    elements = super(WhitespaceSeparatedListParser, self)._custom_xml_dom_elements(doc)
    separators = list(string.whitespace)
    if self._comma_compat:
        separators.append(',')
    separators.sort()
    for sep_char in separators:
        elements.append(_helpers.create_xml_dom_element(doc, 'list_separator', repr(sep_char)))
    return elements