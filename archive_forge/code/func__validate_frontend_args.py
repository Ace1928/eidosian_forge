from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def _validate_frontend_args(parser, lexer) -> None:
    assert_config(parser, ('lalr', 'earley', 'cyk'))
    if not isinstance(lexer, type):
        expected = {'lalr': ('basic', 'contextual'), 'earley': ('basic', 'dynamic', 'dynamic_complete'), 'cyk': ('basic',)}[parser]
        assert_config(lexer, expected, 'Parser %r does not support lexer %%r, expected one of %%s' % parser)