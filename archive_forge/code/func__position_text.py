from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _position_text(self):
    numwords = len(self._sent)
    num_matched = numwords - len(self._parser.remaining_text())
    leaves = self._tree_leaves()[:num_matched]
    xmax = self._tree.bbox()[0]
    for i in range(0, len(leaves)):
        widget = self._textwidgets[i]
        leaf = leaves[i]
        widget['color'] = '#006040'
        leaf['color'] = '#006040'
        widget.move(leaf.bbox()[0] - widget.bbox()[0], 0)
        xmax = widget.bbox()[2] + 10
    for i in range(len(leaves), numwords):
        widget = self._textwidgets[i]
        widget['color'] = '#a0a0a0'
        widget.move(xmax - widget.bbox()[0], 0)
        xmax = widget.bbox()[2] + 10
    if self._parser.currently_complete():
        for twidget in self._textwidgets:
            twidget['color'] = '#00a000'
    for i in range(0, len(leaves)):
        widget = self._textwidgets[i]
        leaf = leaves[i]
        dy = widget.bbox()[1] - leaf.bbox()[3] - 10.0
        dy = max(dy, leaf.parent().label().bbox()[3] - leaf.bbox()[3] + 10)
        leaf.move(0, dy)