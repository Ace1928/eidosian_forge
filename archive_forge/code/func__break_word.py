import math
import os
import re
import textwrap
from itertools import chain, groupby
from typing import (TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from docutils import nodes, writers
from docutils.nodes import Element, Text
from docutils.utils import column_width
from sphinx import addnodes
from sphinx.locale import _, admonitionlabels
from sphinx.util.docutils import SphinxTranslator
def _break_word(self, word: str, space_left: int) -> Tuple[str, str]:
    """_break_word(word : string, space_left : int) -> (string, string)

        Break line by unicode width instead of len(word).
        """
    total = 0
    for i, c in enumerate(word):
        total += column_width(c)
        if total > space_left:
            return (word[:i - 1], word[i - 1:])
    return (word, '')