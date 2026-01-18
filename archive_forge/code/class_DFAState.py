from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
class DFAState(Generic[_TokenTypeT]):
    """
    The DFAState object is the core class for pretty much anything. DFAState
    are the vertices of an ordered graph while arcs and transitions are the
    edges.

    Arcs are the initial edges, where most DFAStates are not connected and
    transitions are then calculated to connect the DFA state machines that have
    different nonterminals.
    """

    def __init__(self, from_rule: str, nfa_set: Set[NFAState], final: NFAState):
        assert isinstance(nfa_set, set)
        assert isinstance(next(iter(nfa_set)), NFAState)
        assert isinstance(final, NFAState)
        self.from_rule = from_rule
        self.nfa_set = nfa_set
        self.arcs: Mapping[str, DFAState] = {}
        self.nonterminal_arcs: Mapping[str, DFAState] = {}
        self.transitions: Mapping[Union[_TokenTypeT, ReservedString], DFAPlan] = {}
        self.is_final = final in nfa_set

    def add_arc(self, next_, label):
        assert isinstance(label, str)
        assert label not in self.arcs
        assert isinstance(next_, DFAState)
        self.arcs[label] = next_

    def unifystate(self, old, new):
        for label, next_ in self.arcs.items():
            if next_ is old:
                self.arcs[label] = new

    def __eq__(self, other):
        assert isinstance(other, DFAState)
        if self.is_final != other.is_final:
            return False
        if len(self.arcs) != len(other.arcs):
            return False
        for label, next_ in self.arcs.items():
            if next_ is not other.arcs.get(label):
                return False
        return True

    def __repr__(self):
        return '<%s: %s is_final=%s>' % (self.__class__.__name__, self.from_rule, self.is_final)