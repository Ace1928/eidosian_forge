import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
def _draw_treecycle(self):
    if len(self._treetoks) <= 1:
        return
    label = '%d Trees' % len(self._treetoks)
    c = self._tree_canvas
    margin = ChartView._MARGIN
    right = self._chart.num_leaves() * self._unitsize + margin - 2
    tag = c.create_text(right, 2, anchor='ne', text=label, font=self._boldfont)
    self._tree_tags.append(tag)
    _, _, _, y = c.bbox(tag)
    for i in range(len(self._treetoks)):
        x = right - 20 * (len(self._treetoks) - i - 1)
        if i == self._treetoks_index:
            fill = '#084'
        else:
            fill = '#fff'
        tag = c.create_polygon(x, y + 10, x - 5, y, x - 10, y + 10, fill=fill, outline='black')
        self._tree_tags.append(tag)

        def cb(event, self=self, i=i):
            self._treetoks_index = i
            self.draw_tree()
        c.tag_bind(tag, '<Button-1>', cb)