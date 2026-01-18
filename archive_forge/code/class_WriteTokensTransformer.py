from typing import Dict, Callable, Iterable, Optional
from .lark import Lark
from .tree import Tree, ParseTree
from .visitors import Transformer_InPlace
from .lexer import Token, PatternStr, TerminalDef
from .grammar import Terminal, NonTerminal, Symbol
from .tree_matcher import TreeMatcher, is_discarded_terminal
from .utils import is_id_continue
class WriteTokensTransformer(Transformer_InPlace):
    """Inserts discarded tokens into their correct place, according to the rules of grammar"""
    tokens: Dict[str, TerminalDef]
    term_subs: Dict[str, Callable[[Symbol], str]]

    def __init__(self, tokens: Dict[str, TerminalDef], term_subs: Dict[str, Callable[[Symbol], str]]) -> None:
        self.tokens = tokens
        self.term_subs = term_subs

    def __default__(self, data, children, meta):
        if not getattr(meta, 'match_tree', False):
            return Tree(data, children)
        iter_args = iter(children)
        to_write = []
        for sym in meta.orig_expansion:
            if is_discarded_terminal(sym):
                try:
                    v = self.term_subs[sym.name](sym)
                except KeyError:
                    t = self.tokens[sym.name]
                    if not isinstance(t.pattern, PatternStr):
                        raise NotImplementedError('Reconstructing regexps not supported yet: %s' % t)
                    v = t.pattern.value
                to_write.append(v)
            else:
                x = next(iter_args)
                if isinstance(x, list):
                    to_write += x
                else:
                    if isinstance(x, Token):
                        assert Terminal(x.type) == sym, x
                    else:
                        assert NonTerminal(x.data) == sym, (sym, x)
                    to_write.append(x)
        assert is_iter_empty(iter_args)
        return to_write