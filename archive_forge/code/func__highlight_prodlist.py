from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _highlight_prodlist(self):
    self._prodlist.delete(0, 'end')
    expandable = self._parser.expandable_productions()
    untried = self._parser.untried_expandable_productions()
    productions = self._productions
    for index in range(len(productions)):
        if productions[index] in expandable:
            if productions[index] in untried:
                self._prodlist.insert(index, ' %s' % productions[index])
            else:
                self._prodlist.insert(index, ' %s (TRIED)' % productions[index])
            self._prodlist.selection_set(index)
        else:
            self._prodlist.insert(index, ' %s' % productions[index])