import os
import typing
import typing as t
import weakref
from collections import ChainMap
from functools import lru_cache
from functools import partial
from functools import reduce
from types import CodeType
from markupsafe import Markup
from . import nodes
from .compiler import CodeGenerator
from .compiler import generate
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .exceptions import TemplateNotFound
from .exceptions import TemplateRuntimeError
from .exceptions import TemplatesNotFound
from .exceptions import TemplateSyntaxError
from .exceptions import UndefinedError
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .parser import Parser
from .runtime import Context
from .runtime import new_context
from .runtime import Undefined
from .utils import _PassArg
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import LRUCache
from .utils import missing
def _environment_config_check(environment: 'Environment') -> 'Environment':
    """Perform a sanity check on the environment."""
    assert issubclass(environment.undefined, Undefined), "'undefined' must be a subclass of 'jinja2.Undefined'."
    assert environment.block_start_string != environment.variable_start_string != environment.comment_start_string, 'block, variable and comment start strings must be different.'
    assert environment.newline_sequence in {'\r', '\r\n', '\n'}, "'newline_sequence' must be one of '\\n', '\\r\\n', or '\\r'."
    return environment