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
def _depart_admonition(self, node: Element) -> None:
    label = admonitionlabels[node.tagname]
    indent = sum(self.stateindent) + len(label)
    if len(self.states[-1]) == 1 and self.states[-1][0][0] == 0 and (MAXWIDTH - indent >= sum((len(s) for s in self.states[-1][0][1]))):
        self.stateindent[-1] += len(label)
        self.end_state(first=label + ': ')
    else:
        self.states[-1].insert(0, (0, [self.nl]))
        self.end_state(first=label + ':')