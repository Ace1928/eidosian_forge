from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_canvas(self, parent):
    self._cframe = CanvasFrame(parent, background='white', width=525, closeenough=10, border=2, relief='sunken')
    self._cframe.pack(expand=1, fill='both', side='top', pady=2)
    canvas = self._canvas = self._cframe.canvas()
    self._stackwidgets = []
    self._rtextwidgets = []
    self._titlebar = canvas.create_rectangle(0, 0, 0, 0, fill='#c0f0f0', outline='black')
    self._exprline = canvas.create_line(0, 0, 0, 0, dash='.')
    self._stacktop = canvas.create_line(0, 0, 0, 0, fill='#408080')
    size = self._size.get() + 4
    self._stacklabel = TextWidget(canvas, 'Stack', color='#004040', font=self._boldfont)
    self._rtextlabel = TextWidget(canvas, 'Remaining Text', color='#004040', font=self._boldfont)
    self._cframe.add_widget(self._stacklabel)
    self._cframe.add_widget(self._rtextlabel)