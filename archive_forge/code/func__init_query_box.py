import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_query_box(self, parent):
    innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
    another = Frame(innerframe, background=self._BACKGROUND_COLOUR)
    self.query_box = Entry(another, width=60)
    self.query_box.pack(side='left', fill='x', pady=25, anchor='center')
    self.search_button = Button(another, text='Search', command=self.search, borderwidth=1, highlightthickness=1)
    self.search_button.pack(side='left', fill='x', pady=25, anchor='center')
    self.query_box.bind('<KeyPress-Return>', self.search_enter_keypress_handler)
    another.pack()
    innerframe.pack(side='top', fill='x', anchor='n')