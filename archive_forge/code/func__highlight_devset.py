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
def _highlight_devset(self, sample=None):
    if sample is None:
        sample = self.devset[self.devset_index:self.devset_index + 1]
    self.devsetbox.tag_remove('true-pos', '1.0', 'end')
    self.devsetbox.tag_remove('false-neg', '1.0', 'end')
    self.devsetbox.tag_remove('false-pos', '1.0', 'end')
    for sentnum, gold_tree in enumerate(sample):
        test_tree = self._chunkparse(gold_tree.leaves())
        gold_chunks = self._chunks(gold_tree)
        test_chunks = self._chunks(test_tree)
        for chunk in gold_chunks.intersection(test_chunks):
            self._color_chunk(sentnum, chunk, 'true-pos')
        for chunk in gold_chunks - test_chunks:
            self._color_chunk(sentnum, chunk, 'false-neg')
        for chunk in test_chunks - gold_chunks:
            self._color_chunk(sentnum, chunk, 'false-pos')