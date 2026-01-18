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
class LarkOptions(Serialize):
    """Specifies the options for Lark

    """
    start: List[str]
    debug: bool
    strict: bool
    transformer: 'Optional[Transformer]'
    propagate_positions: Union[bool, str]
    maybe_placeholders: bool
    cache: Union[bool, str]
    regex: bool
    g_regex_flags: int
    keep_all_tokens: bool
    tree_class: Optional[Callable[[str, List], Any]]
    parser: _ParserArgType
    lexer: _LexerArgType
    ambiguity: 'Literal["auto", "resolve", "explicit", "forest"]'
    postlex: Optional[PostLex]
    priority: 'Optional[Literal["auto", "normal", "invert"]]'
    lexer_callbacks: Dict[str, Callable[[Token], Token]]
    use_bytes: bool
    ordered_sets: bool
    edit_terminals: Optional[Callable[[TerminalDef], TerminalDef]]
    import_paths: 'List[Union[str, Callable[[Union[None, str, PackageResource], str], Tuple[str, str]]]]'
    source_path: Optional[str]
    OPTIONS_DOC = '\n    **===  General Options  ===**\n\n    start\n            The start symbol. Either a string, or a list of strings for multiple possible starts (Default: "start")\n    debug\n            Display debug information and extra warnings. Use only when debugging (Default: ``False``)\n            When used with Earley, it generates a forest graph as "sppf.png", if \'dot\' is installed.\n    strict\n            Throw an exception on any potential ambiguity, including shift/reduce conflicts, and regex collisions.\n    transformer\n            Applies the transformer to every parse tree (equivalent to applying it after the parse, but faster)\n    propagate_positions\n            Propagates positional attributes into the \'meta\' attribute of all tree branches.\n            Sets attributes: (line, column, end_line, end_column, start_pos, end_pos,\n                              container_line, container_column, container_end_line, container_end_column)\n            Accepts ``False``, ``True``, or a callable, which will filter which nodes to ignore when propagating.\n    maybe_placeholders\n            When ``True``, the ``[]`` operator returns ``None`` when not matched.\n            When ``False``,  ``[]`` behaves like the ``?`` operator, and returns no value at all.\n            (default= ``True``)\n    cache\n            Cache the results of the Lark grammar analysis, for x2 to x3 faster loading. LALR only for now.\n\n            - When ``False``, does nothing (default)\n            - When ``True``, caches to a temporary file in the local directory\n            - When given a string, caches to the path pointed by the string\n    regex\n            When True, uses the ``regex`` module instead of the stdlib ``re``.\n    g_regex_flags\n            Flags that are applied to all terminals (both regex and strings)\n    keep_all_tokens\n            Prevent the tree builder from automagically removing "punctuation" tokens (Default: ``False``)\n    tree_class\n            Lark will produce trees comprised of instances of this class instead of the default ``lark.Tree``.\n\n    **=== Algorithm Options ===**\n\n    parser\n            Decides which parser engine to use. Accepts "earley" or "lalr". (Default: "earley").\n            (there is also a "cyk" option for legacy)\n    lexer\n            Decides whether or not to use a lexer stage\n\n            - "auto" (default): Choose for me based on the parser\n            - "basic": Use a basic lexer\n            - "contextual": Stronger lexer (only works with parser="lalr")\n            - "dynamic": Flexible and powerful (only with parser="earley")\n            - "dynamic_complete": Same as dynamic, but tries *every* variation of tokenizing possible.\n    ambiguity\n            Decides how to handle ambiguity in the parse. Only relevant if parser="earley"\n\n            - "resolve": The parser will automatically choose the simplest derivation\n              (it chooses consistently: greedy for tokens, non-greedy for rules)\n            - "explicit": The parser will return all derivations wrapped in "_ambig" tree nodes (i.e. a forest).\n            - "forest": The parser will return the root of the shared packed parse forest.\n\n    **=== Misc. / Domain Specific Options ===**\n\n    postlex\n            Lexer post-processing (Default: ``None``) Only works with the basic and contextual lexers.\n    priority\n            How priorities should be evaluated - "auto", ``None``, "normal", "invert" (Default: "auto")\n    lexer_callbacks\n            Dictionary of callbacks for the lexer. May alter tokens during lexing. Use with caution.\n    use_bytes\n            Accept an input of type ``bytes`` instead of ``str``.\n    ordered_sets\n            Should Earley use ordered-sets to achieve stable output (~10% slower than regular sets. Default: True)\n    edit_terminals\n            A callback for editing the terminals before parse.\n    import_paths\n            A List of either paths or loader functions to specify from where grammars are imported\n    source_path\n            Override the source of from where the grammar was loaded. Useful for relative imports and unconventional grammar loading\n    **=== End of Options ===**\n    '
    if __doc__:
        __doc__ += OPTIONS_DOC
    _defaults: Dict[str, Any] = {'debug': False, 'strict': False, 'keep_all_tokens': False, 'tree_class': None, 'cache': False, 'postlex': None, 'parser': 'earley', 'lexer': 'auto', 'transformer': None, 'start': 'start', 'priority': 'auto', 'ambiguity': 'auto', 'regex': False, 'propagate_positions': False, 'lexer_callbacks': {}, 'maybe_placeholders': True, 'edit_terminals': None, 'g_regex_flags': 0, 'use_bytes': False, 'ordered_sets': True, 'import_paths': [], 'source_path': None, '_plugins': {}}

    def __init__(self, options_dict: Dict[str, Any]) -> None:
        o = dict(options_dict)
        options = {}
        for name, default in self._defaults.items():
            if name in o:
                value = o.pop(name)
                if isinstance(default, bool) and name not in ('cache', 'use_bytes', 'propagate_positions'):
                    value = bool(value)
            else:
                value = default
            options[name] = value
        if isinstance(options['start'], str):
            options['start'] = [options['start']]
        self.__dict__['options'] = options
        assert_config(self.parser, ('earley', 'lalr', 'cyk', None))
        if self.parser == 'earley' and self.transformer:
            raise ConfigurationError('Cannot specify an embedded transformer when using the Earley algorithm. Please use your transformer on the resulting parse tree, or use a different algorithm (i.e. LALR)')
        if o:
            raise ConfigurationError('Unknown options: %s' % o.keys())

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__['options'][name]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name: str, value: str) -> None:
        assert_config(name, self.options.keys(), "%r isn't a valid option. Expected one of: %s")
        self.options[name] = value

    def serialize(self, memo=None) -> Dict[str, Any]:
        return self.options

    @classmethod
    def deserialize(cls, data: Dict[str, Any], memo: Dict[int, Union[TerminalDef, Rule]]) -> 'LarkOptions':
        return cls(data)