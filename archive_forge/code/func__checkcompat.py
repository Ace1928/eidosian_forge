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
def _checkcompat(self):
    if self._left_chart.tokens() != self._right_chart.tokens() or self._left_chart.property_names() != self._right_chart.property_names() or self._left_chart == self._emptychart or (self._right_chart == self._emptychart):
        self._out_chart = self._emptychart
        self._out_matrix.set_chart(self._out_chart)
        self._out_matrix.inactivate()
        self._out_label['text'] = 'Output'
        return False
    else:
        return True