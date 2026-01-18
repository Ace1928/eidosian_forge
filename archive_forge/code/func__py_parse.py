import sys
import types
import antlr4
from antlr4 import InputStream, CommonTokenStream, Token
from antlr4.tree.Tree import ParseTree
from antlr4.error.ErrorListener import ErrorListener
from .fugue_sqlParser import fugue_sqlParser
from .fugue_sqlLexer import fugue_sqlLexer
def _py_parse(stream: InputStream, entry_rule_name: str, sa_err_listener: SA_ErrorListener=None) -> ParseTree:
    if sa_err_listener is not None:
        err_listener = _FallbackErrorTranslator(sa_err_listener, stream)
    lexer = fugue_sqlLexer(stream)
    if sa_err_listener is not None:
        lexer.removeErrorListeners()
        lexer.addErrorListener(err_listener)
    token_stream = CommonTokenStream(lexer)
    parser = fugue_sqlParser(token_stream)
    if sa_err_listener is not None:
        parser.removeErrorListeners()
        parser.addErrorListener(err_listener)
    entry_rule_func = getattr(parser, entry_rule_name, None)
    if not isinstance(entry_rule_func, types.MethodType):
        raise ValueError("Invalid entry_rule_name '%s'" % entry_rule_name)
    return entry_rule_func()