from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _manage_vertical(self):
    nodex, nodey = self._node_bottom()
    x = 0
    for subtree in self._subtrees:
        subtree_bbox = subtree.bbox()
        dy = nodey - subtree_bbox[1] + self._yspace
        dx = x - subtree_bbox[0]
        subtree.move(dx, dy)
        x += subtree_bbox[2] - subtree_bbox[0] + self._xspace
    center = 0.0
    for subtree in self._subtrees:
        center += self._subtree_top(subtree)[0] / len(self._subtrees)
    for subtree in self._subtrees:
        subtree.move(nodex - center, 0)