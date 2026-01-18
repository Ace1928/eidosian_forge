import random
import re
import textwrap
import time
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle
def _devset_scroll(self, command, *args):
    N = 1
    showing_trace = self._showing_trace
    if command == 'scroll' and args[1].startswith('unit'):
        self.show_devset(self.devset_index + int(args[0]))
    elif command == 'scroll' and args[1].startswith('page'):
        self.show_devset(self.devset_index + N * int(args[0]))
    elif command == 'moveto':
        self.show_devset(int(float(args[0]) * self._devset_size.get()))
    else:
        assert 0, f'bad scroll command {command} {args}'
    if showing_trace:
        self.show_trace()