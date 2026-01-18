from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _highlight_productions(self):
    self._prodlist.selection_clear(0, 'end')
    for prod in self._parser.reducible_productions():
        index = self._productions.index(prod)
        self._prodlist.selection_set(index)