from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_shift(self):
    widget = self._rtextwidgets[0]
    right = widget.bbox()[0]
    if len(self._stackwidgets) == 0:
        left = 5
    else:
        left = self._stackwidgets[-1].bbox()[2] + 10
    dt = self._animate.get()
    dx = (left - right) * 1.0 / dt
    self._animate_shift_frame(dt, widget, dx)