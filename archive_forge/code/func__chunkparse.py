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
def _chunkparse(self, words):
    try:
        return self.chunker.parse(words)
    except (ValueError, IndexError) as e:
        self.grammarbox.tag_add('error', '1.0', 'end')
        return words