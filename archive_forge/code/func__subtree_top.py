from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _subtree_top(self, child):
    if isinstance(child, TreeSegmentWidget):
        bbox = child.label().bbox()
    else:
        bbox = child.bbox()
    if self._horizontal:
        return (bbox[0], (bbox[1] + bbox[3]) / 2.0)
    else:
        return ((bbox[0] + bbox[2]) / 2.0, bbox[1])