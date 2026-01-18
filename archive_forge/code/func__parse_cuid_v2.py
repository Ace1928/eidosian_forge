import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _parse_cuid_v2(self, label):
    """Parse a string (v2 repr format) and yield name, idx pairs

        This attempts to parse a string (nominally returned by
        get_repr()) to generate the sequence of (name, idx) pairs for
        the _cuids data structure.

        """
    if ComponentUID._lex is None:
        ComponentUID._lex = ply.lex.lex()
    name = None
    idx_stack = []
    idx = ()
    self._lex.input(label)
    while True:
        tok = self._lex.token()
        if not tok:
            break
        if tok.type == '.':
            assert not idx_stack
            yield (name, idx)
            name = None
            idx = ()
        elif tok.type == '[':
            idx_stack.append([])
        elif tok.type == ']':
            idx = tuple(idx_stack.pop())
            assert not idx_stack
        elif tok.type == '(':
            assert idx_stack
            idx_stack.append([])
        elif tok.type == ')':
            tmp = tuple(idx_stack.pop())
            idx_stack[-1].append(tmp)
        elif idx_stack:
            if tok.type == ',':
                pass
            elif tok.type == 'STAR':
                idx_stack[-1].append(tok.value)
            else:
                assert tok.type in {'WORD', 'STRING', 'NUMBER', 'PICKLE'}
                idx_stack[-1].append(tok.value)
        else:
            assert tok.type in {'WORD', 'STRING'}
            assert name is None
            name = tok.value
    assert not idx_stack
    yield (name, idx)