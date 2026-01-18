from contextlib import contextmanager
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pytree import NL, Context, Leaf, Node, RawNode, convert
from . import grammar, token, tokenize
def _addtoken(self, ilabel: int, type: int, value: str, context: Context) -> bool:
    while True:
        dfa, state, node = self.stack[-1]
        states, first = dfa
        arcs = states[state]
        for i, newstate in arcs:
            t = self.grammar.labels[i][0]
            if t >= 256:
                itsdfa = self.grammar.dfas[t]
                itsstates, itsfirst = itsdfa
                if ilabel in itsfirst:
                    self.push(t, itsdfa, newstate, context)
                    break
            elif ilabel == i:
                self.shift(type, value, newstate, context)
                state = newstate
                while states[state] == [(0, state)]:
                    self.pop()
                    if not self.stack:
                        return True
                    dfa, state, node = self.stack[-1]
                    states, first = dfa
                self.last_token = type
                return False
        else:
            if (0, state) in arcs:
                self.pop()
                if not self.stack:
                    raise ParseError('too much input', type, value, context)
            else:
                raise ParseError('bad input', type, value, context)