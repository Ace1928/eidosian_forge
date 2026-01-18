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
def _animate_strategy(self, speed=1):
    if self._animating == 0:
        return
    if self._apply_strategy() is not None:
        if self._animate.get() == 0 or self._step.get() == 1:
            return
        if self._animate.get() == 1:
            self._root.after(3000, self._animate_strategy)
        elif self._animate.get() == 2:
            self._root.after(1000, self._animate_strategy)
        else:
            self._root.after(20, self._animate_strategy)