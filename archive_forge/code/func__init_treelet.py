import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _init_treelet(self, parent):
    self._treelet_canvas = Canvas(parent, background='white')
    self._treelet_canvas.pack(side='bottom', fill='x')
    self._treelet = None