from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _backtrack(self, *e):
    if self._animating_lock:
        return
    if self._parser.backtrack():
        elt = self._parser.tree()
        for i in self._parser.frontier()[0]:
            elt = elt[i]
        self._lastoper1['text'] = 'Backtrack'
        self._lastoper2['text'] = ''
        if isinstance(elt, Tree):
            self._animate_backtrack(self._parser.frontier()[0])
        else:
            self._animate_match_backtrack(self._parser.frontier()[0])
        return True
    else:
        self._autostep = 0
        self._lastoper1['text'] = 'Finished'
        self._lastoper2['text'] = ''
        return False