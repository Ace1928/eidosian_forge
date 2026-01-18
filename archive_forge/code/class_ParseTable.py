from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
class ParseTable(ParseTableBase['State']):
    """Parse-table whose key is State, i.e. set[RulePtr]

    Slower than IntParseTable, but useful for debugging
    """
    pass