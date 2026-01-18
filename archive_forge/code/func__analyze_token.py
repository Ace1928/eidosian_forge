import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _analyze_token(self, match, linenum):
    """
        Given a line number and a regexp match for a token on that
        line, colorize the token.  Note that the regexp match gives us
        the token's text, start index (on the line), and end index (on
        the line).
        """
    if match.group()[0] in '\'"':
        tag = 'terminal'
    elif match.group() in ('->', self.ARROW):
        tag = 'arrow'
    else:
        tag = 'nonterminal_' + match.group()
        if tag not in self._textwidget.tag_names():
            self._init_nonterminal_tag(tag)
    start = '%d.%d' % (linenum, match.start())
    end = '%d.%d' % (linenum, match.end())
    self._textwidget.tag_add(tag, start, end)