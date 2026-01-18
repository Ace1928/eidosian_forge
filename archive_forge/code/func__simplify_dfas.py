from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _simplify_dfas(dfas):
    """
    This is not theoretically optimal, but works well enough.
    Algorithm: repeatedly look for two states that have the same
    set of arcs (same labels pointing to the same nodes) and
    unify them, until things stop changing.

    dfas is a list of DFAState instances
    """
    changes = True
    while changes:
        changes = False
        for i, state_i in enumerate(dfas):
            for j in range(i + 1, len(dfas)):
                state_j = dfas[j]
                if state_i == state_j:
                    del dfas[j]
                    for state in dfas:
                        state.unifystate(state_j, state_i)
                    changes = True
                    break