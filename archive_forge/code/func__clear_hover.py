from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _clear_hover(self, *event):
    if self._hover == -1:
        return
    self._hover = -1
    for stackwidget in self._stackwidgets:
        if isinstance(stackwidget, TreeSegmentWidget):
            stackwidget.label()['color'] = 'black'
        else:
            stackwidget['color'] = 'black'