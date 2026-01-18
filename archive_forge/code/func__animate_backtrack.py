from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_backtrack(self, treeloc):
    if self._animation_frames.get() == 0:
        colors = []
    else:
        colors = ['#a00000', '#000000', '#a00000']
    colors += ['gray%d' % (10 * int(10 * x / self._animation_frames.get())) for x in range(1, self._animation_frames.get() + 1)]
    widgets = [self._get(self._tree, treeloc).parent()]
    for subtree in widgets[0].subtrees():
        if isinstance(subtree, TreeSegmentWidget):
            widgets.append(subtree.label())
        else:
            widgets.append(subtree)
    self._animate_backtrack_frame(widgets, colors)