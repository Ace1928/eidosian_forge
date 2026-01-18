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
def _reset_parser(self):
    self._cp = SteppingChartParser(self._grammar)
    self._cp.initialize(self._tokens)
    self._chart = self._cp.chart()
    for _new_edge in LeafInitRule().apply(self._chart, self._grammar):
        pass
    self._cpstep = self._cp.step()
    self._selection = None