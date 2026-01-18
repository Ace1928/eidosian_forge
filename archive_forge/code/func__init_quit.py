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
def _init_quit(self, root):
    quit = Button(root, text='Quit', command=self.destroy)
    quit.pack(side='bottom', expand=0, fill='none')