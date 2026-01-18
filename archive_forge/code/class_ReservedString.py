from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
class ReservedString:
    """
    Most grammars will have certain keywords and operators that are mentioned
    in the grammar as strings (e.g. "if") and not token types (e.g. NUMBER).
    This class basically is the former.
    """

    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.value)