import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_status(self, parent):
    self.status = Label(parent, justify=LEFT, relief=SUNKEN, background=self._BACKGROUND_COLOUR, border=0, padx=1, pady=0)
    self.status.pack(side='top', anchor='sw')