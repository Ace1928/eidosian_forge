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
def _init_colortags(self, textwidget, options):
    textwidget.tag_config('terminal', foreground='#006000')
    textwidget.tag_config('arrow', font='symbol', underline='0')
    textwidget.tag_config('dot', foreground='#000000')
    textwidget.tag_config('nonterminal', foreground='blue', font=('helvetica', -12, 'bold'))