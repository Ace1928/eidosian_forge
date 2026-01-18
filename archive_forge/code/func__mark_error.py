import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _mark_error(self, linenum, line):
    """
        Mark the location of an error in a line.
        """
    arrowmatch = CFGEditor._ARROW_RE.search(line)
    if not arrowmatch:
        start = '%d.0' % linenum
        end = '%d.end' % linenum
    elif not CFGEditor._LHS_RE.match(line):
        start = '%d.0' % linenum
        end = '%d.%d' % (linenum, arrowmatch.start())
    else:
        start = '%d.%d' % (linenum, arrowmatch.end())
        end = '%d.end' % linenum
    if self._textwidget.compare(start, '==', end):
        start = '%d.0' % linenum
        end = '%d.end' % linenum
    self._textwidget.tag_add('error', start, end)