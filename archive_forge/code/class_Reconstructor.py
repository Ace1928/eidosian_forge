from typing import Dict, Callable, Iterable, Optional
from .lark import Lark
from .tree import Tree, ParseTree
from .visitors import Transformer_InPlace
from .lexer import Token, PatternStr, TerminalDef
from .grammar import Terminal, NonTerminal, Symbol
from .tree_matcher import TreeMatcher, is_discarded_terminal
from .utils import is_id_continue
class Reconstructor(TreeMatcher):
    """
    A Reconstructor that will, given a full parse Tree, generate source code.

    Note:
        The reconstructor cannot generate values from regexps. If you need to produce discarded
        regexes, such as newlines, use `term_subs` and provide default values for them.

    Parameters:
        parser: a Lark instance
        term_subs: a dictionary of [Terminal name as str] to [output text as str]
    """
    write_tokens: WriteTokensTransformer

    def __init__(self, parser: Lark, term_subs: Optional[Dict[str, Callable[[Symbol], str]]]=None) -> None:
        TreeMatcher.__init__(self, parser)
        self.write_tokens = WriteTokensTransformer({t.name: t for t in self.tokens}, term_subs or {})

    def _reconstruct(self, tree):
        unreduced_tree = self.match_tree(tree, tree.data)
        res = self.write_tokens.transform(unreduced_tree)
        for item in res:
            if isinstance(item, Tree):
                yield from self._reconstruct(item)
            else:
                yield item

    def reconstruct(self, tree: ParseTree, postproc: Optional[Callable[[Iterable[str]], Iterable[str]]]=None, insert_spaces: bool=True) -> str:
        x = self._reconstruct(tree)
        if postproc:
            x = postproc(x)
        y = []
        prev_item = ''
        for item in x:
            if insert_spaces and prev_item and item and is_id_continue(prev_item[-1]) and is_id_continue(item[0]):
                y.append(' ')
            y.append(item)
            prev_item = item
        return ''.join(y)