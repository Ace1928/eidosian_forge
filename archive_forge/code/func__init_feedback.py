from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle
def _init_feedback(self, parent):
    self._feedbackframe = feedbackframe = Frame(parent)
    feedbackframe.pack(fill='x', side='bottom', padx=3, pady=3)
    self._lastoper_label = Label(feedbackframe, text='Last Operation:', font=self._font)
    self._lastoper_label.pack(side='left')
    lastoperframe = Frame(feedbackframe, relief='sunken', border=1)
    lastoperframe.pack(fill='x', side='right', expand=1, padx=5)
    self._lastoper1 = Label(lastoperframe, foreground='#007070', background='#f0f0f0', font=self._font)
    self._lastoper2 = Label(lastoperframe, anchor='w', width=30, foreground='#004040', background='#f0f0f0', font=self._font)
    self._lastoper1.pack(side='left')
    self._lastoper2.pack(side='left', fill='x', expand=1)