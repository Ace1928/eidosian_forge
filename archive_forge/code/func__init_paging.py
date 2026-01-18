import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_paging(self, parent):
    innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
    self.prev = prev = Button(innerframe, text='Previous', command=self.previous, width='10', borderwidth=1, highlightthickness=1, state='disabled')
    prev.pack(side='left', anchor='center')
    self.next = next = Button(innerframe, text='Next', command=self.__next__, width='10', borderwidth=1, highlightthickness=1, state='disabled')
    next.pack(side='right', anchor='center')
    innerframe.pack(side='top', fill='y')
    self.current_page = 0