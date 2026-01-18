from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_match_backtrack_frame(self, frame, widget, dy):
    if frame > 0:
        self._animating_lock = 1
        widget.move(0, dy)
        self._top.after(10, self._animate_match_backtrack_frame, frame - 1, widget, dy)
    else:
        widget.parent().remove_child(widget)
        widget.destroy()
        self._animating_lock = 0
        if self._autostep:
            self._step()