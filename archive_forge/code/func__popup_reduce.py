from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _popup_reduce(self, widget):
    productions = self._parser.reducible_productions()
    if len(productions) == 0:
        return
    self._reduce_menu.delete(0, 'end')
    for production in productions:
        self._reduce_menu.add_command(label=str(production), command=self.reduce)
    self._reduce_menu.post(self._canvas.winfo_pointerx(), self._canvas.winfo_pointery())