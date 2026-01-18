import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
class CFGDemo:

    def __init__(self, grammar, text):
        self._grammar = grammar
        self._text = text
        self._top = Tk()
        self._top.title('Context Free Grammar Demo')
        self._size = IntVar(self._top)
        self._size.set(12)
        self._init_bindings(self._top)
        frame1 = Frame(self._top)
        frame1.pack(side='left', fill='y', expand=0)
        self._init_menubar(self._top)
        self._init_buttons(self._top)
        self._init_grammar(frame1)
        self._init_treelet(frame1)
        self._init_workspace(self._top)

    def _init_bindings(self, top):
        top.bind('<Control-q>', self.destroy)

    def _init_menubar(self, parent):
        pass

    def _init_buttons(self, parent):
        pass

    def _init_grammar(self, parent):
        self._prodlist = ProductionList(parent, self._grammar, width=20)
        self._prodlist.pack(side='top', fill='both', expand=1)
        self._prodlist.focus()
        self._prodlist.add_callback('select', self._selectprod_cb)
        self._prodlist.add_callback('move', self._selectprod_cb)

    def _init_treelet(self, parent):
        self._treelet_canvas = Canvas(parent, background='white')
        self._treelet_canvas.pack(side='bottom', fill='x')
        self._treelet = None

    def _init_workspace(self, parent):
        self._workspace = CanvasFrame(parent, background='white')
        self._workspace.pack(side='right', fill='both', expand=1)
        self._tree = None
        self.reset_workspace()

    def reset_workspace(self):
        c = self._workspace.canvas()
        fontsize = int(self._size.get())
        node_font = ('helvetica', -(fontsize + 4), 'bold')
        leaf_font = ('helvetica', -(fontsize + 2))
        if self._tree is not None:
            self._workspace.remove_widget(self._tree)
        start = self._grammar.start().symbol()
        rootnode = TextWidget(c, start, font=node_font, draggable=1)
        leaves = []
        for word in self._text:
            leaves.append(TextWidget(c, word, font=leaf_font, draggable=1))
        self._tree = TreeSegmentWidget(c, rootnode, leaves, color='white')
        self._workspace.add_widget(self._tree)
        for leaf in leaves:
            leaf.move(0, 100)

    def workspace_markprod(self, production):
        pass

    def _markproduction(self, prod, tree=None):
        if tree is None:
            tree = self._tree
        for i in range(len(tree.subtrees()) - len(prod.rhs())):
            if tree['color', i] == 'white':
                self._markproduction
            for j, node in enumerate(prod.rhs()):
                widget = tree.subtrees()[i + j]
                if isinstance(node, Nonterminal) and isinstance(widget, TreeSegmentWidget) and (node.symbol == widget.label().text()):
                    pass
                elif isinstance(node, str) and isinstance(widget, TextWidget) and (node == widget.text()):
                    pass
                else:
                    break
            else:
                print('MATCH AT', i)

    def _selectprod_cb(self, production):
        canvas = self._treelet_canvas
        self._prodlist.highlight(production)
        if self._treelet is not None:
            self._treelet.destroy()
        rhs = production.rhs()
        for i, elt in enumerate(rhs):
            if isinstance(elt, Nonterminal):
                elt = Tree(elt)
        tree = Tree(production.lhs().symbol(), *rhs)
        fontsize = int(self._size.get())
        node_font = ('helvetica', -(fontsize + 4), 'bold')
        leaf_font = ('helvetica', -(fontsize + 2))
        self._treelet = tree_to_treesegment(canvas, tree, node_font=node_font, leaf_font=leaf_font)
        self._treelet['draggable'] = 1
        x1, y1, x2, y2 = self._treelet.bbox()
        w, h = (int(canvas['width']), int(canvas['height']))
        self._treelet.move((w - x1 - x2) / 2, (h - y1 - y2) / 2)
        self._markproduction(production)

    def destroy(self, *args):
        self._top.destroy()

    def mainloop(self, *args, **kwargs):
        self._top.mainloop(*args, **kwargs)