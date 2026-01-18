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
def _eval_plot(self, *e, **config):
    width = config.get('width', self.evalbox.winfo_width())
    height = config.get('height', self.evalbox.winfo_height())
    self.evalbox.delete('all')
    tag = self.evalbox.create_text(10, height // 2 - 10, justify='left', anchor='w', text='Precision')
    left, right = (self.evalbox.bbox(tag)[2] + 5, width - 10)
    tag = self.evalbox.create_text(left + (width - left) // 2, height - 10, anchor='s', text='Recall', justify='center')
    top, bot = (10, self.evalbox.bbox(tag)[1] - 10)
    bg = self._EVALBOX_PARAMS['background']
    self.evalbox.lower(self.evalbox.create_rectangle(0, 0, left - 1, 5000, fill=bg, outline=bg))
    self.evalbox.lower(self.evalbox.create_rectangle(0, bot + 1, 5000, 5000, fill=bg, outline=bg))
    if self._autoscale.get() and len(self._history) > 1:
        max_precision = max_recall = 0
        min_precision = min_recall = 1
        for i in range(1, min(len(self._history), self._SCALE_N + 1)):
            grammar, precision, recall, fmeasure = self._history[-i]
            min_precision = min(precision, min_precision)
            min_recall = min(recall, min_recall)
            max_precision = max(precision, max_precision)
            max_recall = max(recall, max_recall)
        min_precision = max(min_precision - 0.01, 0)
        min_recall = max(min_recall - 0.01, 0)
        max_precision = min(max_precision + 0.01, 1)
        max_recall = min(max_recall + 0.01, 1)
    else:
        min_precision = min_recall = 0
        max_precision = max_recall = 1
    for i in range(11):
        x = left + (right - left) * ((i / 10.0 - min_recall) / (max_recall - min_recall))
        y = bot - (bot - top) * ((i / 10.0 - min_precision) / (max_precision - min_precision))
        if left < x < right:
            self.evalbox.create_line(x, top, x, bot, fill='#888')
        if top < y < bot:
            self.evalbox.create_line(left, y, right, y, fill='#888')
    self.evalbox.create_line(left, top, left, bot)
    self.evalbox.create_line(left, bot, right, bot)
    self.evalbox.create_text(left - 3, bot, justify='right', anchor='se', text='%d%%' % (100 * min_precision))
    self.evalbox.create_text(left - 3, top, justify='right', anchor='ne', text='%d%%' % (100 * max_precision))
    self.evalbox.create_text(left, bot + 3, justify='center', anchor='nw', text='%d%%' % (100 * min_recall))
    self.evalbox.create_text(right, bot + 3, justify='center', anchor='ne', text='%d%%' % (100 * max_recall))
    prev_x = prev_y = None
    for i, (_, precision, recall, fscore) in enumerate(self._history):
        x = left + (right - left) * ((recall - min_recall) / (max_recall - min_recall))
        y = bot - (bot - top) * ((precision - min_precision) / (max_precision - min_precision))
        if i == self._history_index:
            self.evalbox.create_oval(x - 2, y - 2, x + 2, y + 2, fill='#0f0', outline='#000')
            self.status['text'] = 'Precision: %.2f%%\t' % (precision * 100) + 'Recall: %.2f%%\t' % (recall * 100) + 'F-score: %.2f%%' % (fscore * 100)
        else:
            self.evalbox.lower(self.evalbox.create_oval(x - 2, y - 2, x + 2, y + 2, fill='#afa', outline='#8c8'))
        if prev_x is not None and self._eval_lines.get():
            self.evalbox.lower(self.evalbox.create_line(prev_x, prev_y, x, y, fill='#8c8'))
        prev_x, prev_y = (x, y)