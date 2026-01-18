from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _manage_horizontal(self):
    nodex, nodey = self._node_bottom()
    y = 20
    for subtree in self._subtrees:
        subtree_bbox = subtree.bbox()
        dx = nodex - subtree_bbox[0] + self._xspace
        dy = y - subtree_bbox[1]
        subtree.move(dx, dy)
        y += subtree_bbox[3] - subtree_bbox[1] + self._yspace
    center = 0.0
    for subtree in self._subtrees:
        center += self._subtree_top(subtree)[1]
    center /= len(self._subtrees)
    for subtree in self._subtrees:
        subtree.move(0, nodey - center)