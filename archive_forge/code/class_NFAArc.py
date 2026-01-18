from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
class NFAArc:

    def __init__(self, next_: 'NFAState', nonterminal_or_string: Optional[str]):
        self.next: NFAState = next_
        self.nonterminal_or_string: Optional[str] = nonterminal_or_string

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.nonterminal_or_string)