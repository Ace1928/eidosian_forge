from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _highlight_nodes(self):
    bold = ('helvetica', -self._size.get(), 'bold')
    for treeloc in self._parser.frontier()[:1]:
        self._get(self._tree, treeloc)['color'] = '#20a050'
        self._get(self._tree, treeloc)['font'] = bold
    for treeloc in self._parser.frontier()[1:]:
        self._get(self._tree, treeloc)['color'] = '#008080'