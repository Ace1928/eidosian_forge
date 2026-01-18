from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _prodlist_select(self, event):
    selection = self._prodlist.curselection()
    if len(selection) != 1:
        return
    index = int(selection[0])
    production = self._parser.reduce(self._productions[index])
    if production:
        self._lastoper1['text'] = 'Reduce:'
        self._lastoper2['text'] = '%s' % production
        if self._animate.get():
            self._animate_reduce()
        else:
            self._redraw()
    else:
        self._prodlist.selection_clear(0, 'end')
        for prod in self._parser.reducible_productions():
            index = self._productions.index(prod)
            self._prodlist.selection_set(index)