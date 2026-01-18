import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
class RubyParenthesisString(NavigableString):
    """A NavigableString representing the contents of the <rp> HTML
    element.

    https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rp-element
    """
    pass