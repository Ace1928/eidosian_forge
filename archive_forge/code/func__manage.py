from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _manage(self):
    segs = list(self._expanded_trees.values()) + list(self._collapsed_trees.values())
    for tseg in segs:
        if tseg.hidden():
            tseg.show()
            tseg.manage()
            tseg.hide()