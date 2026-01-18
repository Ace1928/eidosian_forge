from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class WhitespaceSeparatedListParser(BaseListParser):
    """Parser for a whitespace-separated list of strings."""

    def __init__(self, comma_compat=False):
        """Initializer.

    Args:
      comma_compat: bool, whether to support comma as an additional separator.
          If False then only whitespace is supported.  This is intended only for
          backwards compatibility with flags that used to be comma-separated.
    """
        self._comma_compat = comma_compat
        name = 'whitespace or comma' if self._comma_compat else 'whitespace'
        super(WhitespaceSeparatedListParser, self).__init__(None, name)

    def parse(self, argument):
        """Parses argument as whitespace-separated list of strings.

    It also parses argument as comma-separated list of strings if requested.

    Args:
      argument: string argument passed in the commandline.

    Returns:
      [str], the parsed flag value.
    """
        if isinstance(argument, list):
            return argument
        elif not argument:
            return []
        else:
            if self._comma_compat:
                argument = argument.replace(',', ' ')
            return argument.split()

    def _custom_xml_dom_elements(self, doc):
        elements = super(WhitespaceSeparatedListParser, self)._custom_xml_dom_elements(doc)
        separators = list(string.whitespace)
        if self._comma_compat:
            separators.append(',')
        separators.sort()
        for sep_char in separators:
            elements.append(_helpers.create_xml_dom_element(doc, 'list_separator', repr(sep_char)))
        return elements