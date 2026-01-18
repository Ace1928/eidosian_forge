from tkinter import IntVar, Menu, Tk
from nltk.draw.util import (
from nltk.tree import Tree
from nltk.util import in_idle
from a Tree, using non-default widget
def _maintain_order_horizontal(self, child):
    left, top, right, bot = child.bbox()
    if child is self._label:
        for subtree in self._subtrees:
            x1, y1, x2, y2 = subtree.bbox()
            if right + self._xspace > x1:
                subtree.move(right + self._xspace - x1)
        return self._subtrees
    else:
        moved = [child]
        index = self._subtrees.index(child)
        y = bot + self._yspace
        for i in range(index + 1, len(self._subtrees)):
            x1, y1, x2, y2 = self._subtrees[i].bbox()
            if y > y1:
                self._subtrees[i].move(0, y - y1)
                y += y2 - y1 + self._yspace
                moved.append(self._subtrees[i])
        y = top - self._yspace
        for i in range(index - 1, -1, -1):
            x1, y1, x2, y2 = self._subtrees[i].bbox()
            if y < y2:
                self._subtrees[i].move(0, y - y2)
                y -= y2 - y1 + self._yspace
                moved.append(self._subtrees[i])
        x1, y1, x2, y2 = self._label.bbox()
        if x2 > left - self._xspace:
            self._label.move(left - self._xspace - x2, 0)
            moved = self._subtrees
    return moved