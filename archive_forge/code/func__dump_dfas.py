from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _dump_dfas(dfas):
    print('Dump of DFA for', dfas[0].from_rule)
    for i, state in enumerate(dfas):
        print('  State', i, state.is_final and '(final)' or '')
        for nonterminal, next_ in state.arcs.items():
            print('    %s -> %d' % (nonterminal, dfas.index(next_)))