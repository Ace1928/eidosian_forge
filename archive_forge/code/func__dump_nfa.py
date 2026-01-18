from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _dump_nfa(start, finish):
    print('Dump of NFA for', start.from_rule)
    todo = [start]
    for i, state in enumerate(todo):
        print('  State', i, state is finish and '(final)' or '')
        for arc in state.arcs:
            label, next_ = (arc.nonterminal_or_string, arc.next)
            if next_ in todo:
                j = todo.index(next_)
            else:
                j = len(todo)
                todo.append(next_)
            if label is None:
                print('    -> %d' % j)
            else:
                print('    %s -> %d' % (label, j))