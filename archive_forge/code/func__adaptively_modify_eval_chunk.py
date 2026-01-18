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
def _adaptively_modify_eval_chunk(self, t):
    """
        Modify _EVAL_CHUNK to try to keep the amount of time that the
        eval demon takes between _EVAL_DEMON_MIN and _EVAL_DEMON_MAX.

        :param t: The amount of time that the eval demon took.
        """
    if t > self._EVAL_DEMON_MAX and self._EVAL_CHUNK > 5:
        self._EVAL_CHUNK = min(self._EVAL_CHUNK - 1, max(int(self._EVAL_CHUNK * (self._EVAL_DEMON_MAX / t)), self._EVAL_CHUNK - 10))
    elif t < self._EVAL_DEMON_MIN:
        self._EVAL_CHUNK = max(self._EVAL_CHUNK + 1, min(int(self._EVAL_CHUNK * (self._EVAL_DEMON_MIN / t)), self._EVAL_CHUNK + 10))