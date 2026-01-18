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
def _sb_canvas(self, root, expand='y', fill='both', side='bottom'):
    """
        Helper for __init__: construct a canvas with a scrollbar.
        """
    cframe = Frame(root, relief='sunk', border=2)
    cframe.pack(fill=fill, expand=expand, side=side)
    canvas = Canvas(cframe, background='#e0e0e0')
    sb = Scrollbar(cframe, orient='vertical')
    sb.pack(side='right', fill='y')
    canvas.pack(side='left', fill=fill, expand='yes')
    sb['command'] = canvas.yview
    canvas['yscrollcommand'] = sb.set
    return (sb, canvas)