from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _make_dfas(start, finish):
    """
    Uses the powerset construction algorithm to create DFA states from sets of
    NFA states.

    Also does state reduction if some states are not needed.
    """
    assert isinstance(start, NFAState)
    assert isinstance(finish, NFAState)

    def addclosure(nfa_state, base_nfa_set):
        assert isinstance(nfa_state, NFAState)
        if nfa_state in base_nfa_set:
            return
        base_nfa_set.add(nfa_state)
        for nfa_arc in nfa_state.arcs:
            if nfa_arc.nonterminal_or_string is None:
                addclosure(nfa_arc.next, base_nfa_set)
    base_nfa_set = set()
    addclosure(start, base_nfa_set)
    states = [DFAState(start.from_rule, base_nfa_set, finish)]
    for state in states:
        arcs = {}
        for nfa_state in state.nfa_set:
            for nfa_arc in nfa_state.arcs:
                if nfa_arc.nonterminal_or_string is not None:
                    nfa_set = arcs.setdefault(nfa_arc.nonterminal_or_string, set())
                    addclosure(nfa_arc.next, nfa_set)
        for nonterminal_or_string, nfa_set in arcs.items():
            for nested_state in states:
                if nested_state.nfa_set == nfa_set:
                    break
            else:
                nested_state = DFAState(start.from_rule, nfa_set, finish)
                states.append(nested_state)
            state.add_arc(nested_state, nonterminal_or_string)
    return states