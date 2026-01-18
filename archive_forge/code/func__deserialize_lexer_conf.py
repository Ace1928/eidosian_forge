from abc import ABC, abstractmethod
import getpass
import sys, os, pickle
import tempfile
import types
import re
from typing import (
from .exceptions import ConfigurationError, assert_config, UnexpectedInput
from .utils import Serialize, SerializeMemoizer, FS, isascii, logger
from .load_grammar import load_grammar, FromPackageLoader, Grammar, verify_used_files, PackageResource, sha256_digest
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
from .lexer import Lexer, BasicLexer, TerminalDef, LexerThread, Token
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import _validate_frontend_args, _get_lexer_callbacks, _deserialize_parsing_frontend, _construct_parsing_frontend
from .grammar import Rule
def _deserialize_lexer_conf(self, data: Dict[str, Any], memo: Dict[int, Union[TerminalDef, Rule]], options: LarkOptions) -> LexerConf:
    lexer_conf = LexerConf.deserialize(data['lexer_conf'], memo)
    lexer_conf.callbacks = options.lexer_callbacks or {}
    lexer_conf.re_module = regex if options.regex else re
    lexer_conf.use_bytes = options.use_bytes
    lexer_conf.g_regex_flags = options.g_regex_flags
    lexer_conf.skip_validation = True
    lexer_conf.postlex = options.postlex
    return lexer_conf