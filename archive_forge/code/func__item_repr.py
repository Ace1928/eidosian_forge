import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
def _item_repr(self, item):
    contents = []
    contents.append(('%s\t' % item.lhs(), 'nonterminal'))
    contents.append((self.ARROW, 'arrow'))
    for i, elt in enumerate(item.rhs()):
        if i == item.dot():
            contents.append((' *', 'dot'))
        if isinstance(elt, Nonterminal):
            contents.append((' %s' % elt.symbol(), 'nonterminal'))
        else:
            contents.append((' %r' % elt, 'terminal'))
    if item.is_complete():
        contents.append((' *', 'dot'))
    return contents