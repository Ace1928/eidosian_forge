import re
from tkinter import (
from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree
def _init_prodframe(self):
    self._prodframe = Frame(self._top)
    self._textwidget = Text(self._prodframe, background='#e0e0e0', exportselection=1)
    self._textscroll = Scrollbar(self._prodframe, takefocus=0, orient='vertical')
    self._textwidget.config(yscrollcommand=self._textscroll.set)
    self._textscroll.config(command=self._textwidget.yview)
    self._textscroll.pack(side='right', fill='y')
    self._textwidget.pack(expand=1, fill='both', side='left')
    self._textwidget.tag_config('terminal', foreground='#006000')
    self._textwidget.tag_config('arrow', font='symbol')
    self._textwidget.tag_config('error', background='red')
    self._linenum = 0
    self._top.bind('>', self._replace_arrows)
    self._top.bind('<<Paste>>', self._analyze)
    self._top.bind('<KeyPress>', self._check_analyze)
    self._top.bind('<ButtonPress>', self._check_analyze)

    def cycle(e, textwidget=self._textwidget):
        textwidget.tk_focusNext().focus()
    self._textwidget.bind('<Tab>', cycle)
    prod_tuples = [(p.lhs(), [p.rhs()]) for p in self._cfg.productions()]
    for i in range(len(prod_tuples) - 1, 0, -1):
        if prod_tuples[i][0] == prod_tuples[i - 1][0]:
            if () in prod_tuples[i][1]:
                continue
            if () in prod_tuples[i - 1][1]:
                continue
            print(prod_tuples[i - 1][1])
            print(prod_tuples[i][1])
            prod_tuples[i - 1][1].extend(prod_tuples[i][1])
            del prod_tuples[i]
    for lhs, rhss in prod_tuples:
        print(lhs, rhss)
        s = '%s ->' % lhs
        for rhs in rhss:
            for elt in rhs:
                if isinstance(elt, Nonterminal):
                    s += ' %s' % elt
                else:
                    s += ' %r' % elt
            s += ' |'
        s = s[:-2] + '\n'
        self._textwidget.insert('end', s)
    self._analyze()