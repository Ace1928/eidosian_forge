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
def _ensure_has_line(self, line: int) -> None:
    while len(self.lines) < line:
        self.lines.append([])