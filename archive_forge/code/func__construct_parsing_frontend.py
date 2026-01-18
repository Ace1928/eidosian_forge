from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def _construct_parsing_frontend(parser_type: _ParserArgType, lexer_type: _LexerArgType, lexer_conf, parser_conf, options):
    assert isinstance(lexer_conf, LexerConf)
    assert isinstance(parser_conf, ParserConf)
    parser_conf.parser_type = parser_type
    lexer_conf.lexer_type = lexer_type
    return ParsingFrontend(lexer_conf, parser_conf, options)