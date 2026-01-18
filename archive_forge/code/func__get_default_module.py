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
@internalcode
def _get_default_module(self, ctx: t.Optional[Context]=None) -> 'TemplateModule':
    """If a context is passed in, this means that the template was
        imported. Imported templates have access to the current
        template's globals by default, but they can only be accessed via
        the context during runtime.

        If there are new globals, we need to create a new module because
        the cached module is already rendered and will not have access
        to globals from the current context. This new module is not
        cached because the template can be imported elsewhere, and it
        should have access to only the current template's globals.
        """
    if self.environment.is_async:
        raise RuntimeError('Module is not available in async mode.')
    if ctx is not None:
        keys = ctx.globals_keys - self.globals.keys()
        if keys:
            return self.make_module({k: ctx.parent[k] for k in keys})
    if self._module is None:
        self._module = self.make_module()
    return self._module