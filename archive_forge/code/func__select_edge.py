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
def _select_edge(self, edge):
    self._selection = edge
    self._cv.markonly_edge(edge, '#f00')
    self._cv.draw_tree(edge)
    if self._matrix:
        self._matrix.markonly_edge(edge)
    if self._matrix:
        self._matrix.view_edge(edge)