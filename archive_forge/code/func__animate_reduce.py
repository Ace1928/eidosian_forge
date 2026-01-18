from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_reduce(self):
    numwidgets = len(self._parser.stack()[-1])
    widgets = self._stackwidgets[-numwidgets:]
    if isinstance(widgets[0], TreeSegmentWidget):
        ydist = 15 + widgets[0].label().height()
    else:
        ydist = 15 + widgets[0].height()
    dt = self._animate.get()
    dy = ydist * 2.0 / dt
    self._animate_reduce_frame(dt / 2, widgets, dy)