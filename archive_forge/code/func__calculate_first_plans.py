from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _calculate_first_plans(nonterminal_to_dfas, first_plans, nonterminal):
    """
    Calculates the first plan in the first_plans dictionary for every given
    nonterminal. This is going to be used to know when to create stack nodes.
    """
    dfas = nonterminal_to_dfas[nonterminal]
    new_first_plans = {}
    first_plans[nonterminal] = None
    state = dfas[0]
    for transition, next_ in state.transitions.items():
        new_first_plans[transition] = [next_.next_dfa]
    for nonterminal2, next_ in state.nonterminal_arcs.items():
        try:
            first_plans2 = first_plans[nonterminal2]
        except KeyError:
            first_plans2 = _calculate_first_plans(nonterminal_to_dfas, first_plans, nonterminal2)
        else:
            if first_plans2 is None:
                raise ValueError('left recursion for rule %r' % nonterminal)
        for t, pushes in first_plans2.items():
            new_first_plans[t] = [next_] + pushes
    first_plans[nonterminal] = new_first_plans
    return new_first_plans