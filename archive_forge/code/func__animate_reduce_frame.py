from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_reduce_frame(self, frame, widgets, dy):
    if frame > 0:
        self._animating_lock = 1
        for widget in widgets:
            widget.move(0, dy)
        self._top.after(10, self._animate_reduce_frame, frame - 1, widgets, dy)
    else:
        del self._stackwidgets[-len(widgets):]
        for widget in widgets:
            self._cframe.remove_widget(widget)
        tok = self._parser.stack()[-1]
        if not isinstance(tok, Tree):
            raise ValueError()
        label = TextWidget(self._canvas, str(tok.label()), color='#006060', font=self._boldfont)
        widget = TreeSegmentWidget(self._canvas, label, widgets, width=2)
        x1, y1, x2, y2 = self._stacklabel.bbox()
        y = y2 - y1 + 10
        if not self._stackwidgets:
            x = 5
        else:
            x = self._stackwidgets[-1].bbox()[2] + 10
        self._cframe.add_widget(widget, x, y)
        self._stackwidgets.append(widget)
        self._draw_stack_top(widget)
        self._highlight_productions()
        self._animating_lock = 0