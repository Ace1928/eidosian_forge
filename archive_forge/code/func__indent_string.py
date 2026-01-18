import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def _indent_string(self, s, indent_level, formatter, indent_before, indent_after):
    """Add indentation whitespace before and/or after a string.

        :param s: The string to amend with whitespace.
        :param indent_level: The indentation level; affects how much
           whitespace goes before the string.
        :param indent_before: Whether or not to add whitespace
           before the string.
        :param indent_after: Whether or not to add whitespace
           (a newline) after the string.
        """
    space_before = ''
    if indent_before and indent_level:
        space_before = formatter.indent * indent_level
    space_after = ''
    if indent_after:
        space_after = '\n'
    return space_before + s + space_after