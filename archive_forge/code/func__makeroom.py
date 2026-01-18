from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _makeroom(self, treeseg):
    """
        Make sure that no sibling tree bbox's overlap.
        """
    parent = treeseg.parent()
    if not isinstance(parent, TreeSegmentWidget):
        return
    index = parent.subtrees().index(treeseg)
    rsiblings = parent.subtrees()[index + 1:]
    if rsiblings:
        dx = treeseg.bbox()[2] - rsiblings[0].bbox()[0] + 10
        for sibling in rsiblings:
            sibling.move(dx, 0)
    if index > 0:
        lsibling = parent.subtrees()[index - 1]
        dx = max(0, lsibling.bbox()[2] - treeseg.bbox()[0] + 10)
        treeseg.move(dx, 0)
    self._makeroom(parent)