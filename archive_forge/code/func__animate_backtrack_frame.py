from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_backtrack_frame(self, widgets, colors):
    if len(colors) > 0:
        self._animating_lock = 1
        for widget in widgets:
            widget['color'] = colors[0]
        self._top.after(50, self._animate_backtrack_frame, widgets, colors[1:])
    else:
        for widget in widgets[0].subtrees():
            widgets[0].remove_child(widget)
            widget.destroy()
        self._redraw_quick()
        self._animating_lock = 0
        if self._autostep:
            self._step()