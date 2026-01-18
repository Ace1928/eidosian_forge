from typing import Iterable, List, Tuple
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode, Token, Tree
from _qpd_antlr import QPDLexer, QPDParser
def _to_tokens(node: Tree) -> Iterable[Token]:
    if isinstance(node, TerminalNode):
        yield node.getSymbol()
    else:
        for i in range(node.getChildCount()):
            for x in _to_tokens(node.getChild(i)):
                yield x