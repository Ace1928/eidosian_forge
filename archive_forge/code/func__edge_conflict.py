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
def _edge_conflict(self, edge, lvl):
    """
        Return True if the given edge overlaps with any edge on the given
        level.  This is used by _add_edge to figure out what level a
        new edge should be added to.
        """
    s1, e1 = edge.span()
    for otheredge in self._edgelevels[lvl]:
        s2, e2 = otheredge.span()
        if s1 <= s2 < e1 or s2 <= s1 < e2 or s1 == s2 == e1 == e2:
            return True
    return False