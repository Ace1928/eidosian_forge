import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _check_analyze(self, *e):
    """
        Check if we've moved to a new line.  If we have, then remove
        all colorization from the line we moved to, and re-colorize
        the line that we moved from.
        """
    linenum = int(self._textwidget.index('insert').split('.')[0])
    if linenum != self._linenum:
        self._clear_tags(linenum)
        self._analyze_line(self._linenum)
        self._linenum = linenum