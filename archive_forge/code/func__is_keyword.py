from typing import Iterable, List, Tuple
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode, Token, Tree
from _qpd_antlr import QPDLexer, QPDParser
def _is_keyword(token: Token):
    if not hasattr(QPDParser, token.text):
        return False
    return getattr(QPDParser, token.text) == token.type