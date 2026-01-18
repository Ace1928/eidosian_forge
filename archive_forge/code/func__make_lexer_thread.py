from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def _make_lexer_thread(self, text: str) -> Union[str, LexerThread]:
    cls = self.options and self.options._plugins.get('LexerThread') or LexerThread
    return text if self.skip_lexer else cls.from_text(self.lexer, text)