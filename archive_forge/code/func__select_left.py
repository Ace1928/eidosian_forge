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
def _select_left(self, name):
    self._left_name = name
    self._left_chart = self._charts[name]
    self._left_matrix.set_chart(self._left_chart)
    if name == 'None':
        self._left_matrix.inactivate()
    self._apply_op()