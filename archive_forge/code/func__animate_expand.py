from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle
def _animate_expand(self, treeloc):
    oldwidget = self._get(self._tree, treeloc)
    oldtree = oldwidget.parent()
    top = not isinstance(oldtree.parent(), TreeSegmentWidget)
    tree = self._parser.tree()
    for i in treeloc:
        tree = tree[i]
    widget = tree_to_treesegment(self._canvas, tree, node_font=self._boldfont, leaf_color='white', tree_width=2, tree_color='white', node_color='white', leaf_font=self._font)
    widget.label()['color'] = '#20a050'
    oldx, oldy = oldtree.label().bbox()[:2]
    newx, newy = widget.label().bbox()[:2]
    widget.move(oldx - newx, oldy - newy)
    if top:
        self._cframe.add_widget(widget, 0, 5)
        widget.move(30 - widget.label().bbox()[0], 0)
        self._tree = widget
    else:
        oldtree.parent().replace_child(oldtree, widget)
    if widget.subtrees():
        dx = oldx + widget.label().width() / 2 - widget.subtrees()[0].bbox()[0] / 2 - widget.subtrees()[0].bbox()[2] / 2
        for subtree in widget.subtrees():
            subtree.move(dx, 0)
    self._makeroom(widget)
    if top:
        self._cframe.destroy_widget(oldtree)
    else:
        oldtree.destroy()
    colors = ['gray%d' % (10 * int(10 * x / self._animation_frames.get())) for x in range(self._animation_frames.get(), 0, -1)]
    dy = widget.bbox()[3] + 30 - self._canvas.coords(self._textline)[1]
    if dy > 0:
        for twidget in self._textwidgets:
            twidget.move(0, dy)
        self._canvas.move(self._textline, 0, dy)
    self._animate_expand_frame(widget, colors)