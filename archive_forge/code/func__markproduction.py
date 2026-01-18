import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _markproduction(self, prod, tree=None):
    if tree is None:
        tree = self._tree
    for i in range(len(tree.subtrees()) - len(prod.rhs())):
        if tree['color', i] == 'white':
            self._markproduction
        for j, node in enumerate(prod.rhs()):
            widget = tree.subtrees()[i + j]
            if isinstance(node, Nonterminal) and isinstance(widget, TreeSegmentWidget) and (node.symbol == widget.label().text()):
                pass
            elif isinstance(node, str) and isinstance(widget, TextWidget) and (node == widget.text()):
                pass
            else:
                break
        else:
            print('MATCH AT', i)