import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _init_workspace(self, parent):
    self._workspace = CanvasFrame(parent, background='white')
    self._workspace.pack(side='right', fill='both', expand=1)
    self._tree = None
    self.reset_workspace()