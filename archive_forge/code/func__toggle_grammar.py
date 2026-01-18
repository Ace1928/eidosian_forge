from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _toggle_grammar(self, *e):
    if self._show_grammar.get():
        self._prodframe.pack(fill='both', side='left', padx=2, after=self._feedbackframe)
        self._lastoper1['text'] = 'Show Grammar'
    else:
        self._prodframe.pack_forget()
        self._lastoper1['text'] = 'Hide Grammar'
    self._lastoper2['text'] = ''