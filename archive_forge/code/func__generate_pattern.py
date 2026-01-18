from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..util import AtomicString
import re
import xml.etree.ElementTree as etree
def _generate_pattern(self, text: str) -> str:
    """ Given a string, returns a regex pattern to match that string. """
    return f'(?P<abbr>\\b{re.escape(text)}\\b)'