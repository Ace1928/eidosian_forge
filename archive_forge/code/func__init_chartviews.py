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
def _init_chartviews(self, root):
    opfont = ('symbol', -36)
    eqfont = ('helvetica', -36)
    frame = Frame(root, background='#c0c0c0')
    frame.pack(side='top', expand=1, fill='both')
    cv1_frame = Frame(frame, border=3, relief='groove')
    cv1_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
    self._left_selector = MutableOptionMenu(cv1_frame, list(self._charts.keys()), command=self._select_left)
    self._left_selector.pack(side='top', pady=5, fill='x')
    self._left_matrix = ChartMatrixView(cv1_frame, self._emptychart, toplevel=False, show_numedges=True)
    self._left_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
    self._left_matrix.add_callback('select', self.select_edge)
    self._left_matrix.add_callback('select_cell', self.select_cell)
    self._left_matrix.inactivate()
    self._op_label = Label(frame, text=' ', width=3, background='#c0c0c0', font=opfont)
    self._op_label.pack(side='left', padx=5, pady=5)
    cv2_frame = Frame(frame, border=3, relief='groove')
    cv2_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
    self._right_selector = MutableOptionMenu(cv2_frame, list(self._charts.keys()), command=self._select_right)
    self._right_selector.pack(side='top', pady=5, fill='x')
    self._right_matrix = ChartMatrixView(cv2_frame, self._emptychart, toplevel=False, show_numedges=True)
    self._right_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
    self._right_matrix.add_callback('select', self.select_edge)
    self._right_matrix.add_callback('select_cell', self.select_cell)
    self._right_matrix.inactivate()
    Label(frame, text='=', width=3, background='#c0c0c0', font=eqfont).pack(side='left', padx=5, pady=5)
    out_frame = Frame(frame, border=3, relief='groove')
    out_frame.pack(side='left', padx=8, pady=7, expand=1, fill='both')
    self._out_label = Label(out_frame, text='Output')
    self._out_label.pack(side='top', pady=9)
    self._out_matrix = ChartMatrixView(out_frame, self._emptychart, toplevel=False, show_numedges=True)
    self._out_matrix.pack(side='bottom', padx=5, pady=5, expand=1, fill='both')
    self._out_matrix.add_callback('select', self.select_edge)
    self._out_matrix.add_callback('select_cell', self.select_cell)
    self._out_matrix.inactivate()