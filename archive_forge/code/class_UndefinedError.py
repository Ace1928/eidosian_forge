from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class UndefinedError(TemplateRuntimeError):
    """Exception thrown when a template expression attempts to access a variable
    not defined in the context.
    
    :see: `LenientLookup`, `StrictLookup`
    """

    def __init__(self, name, owner=UNDEFINED):
        if owner is not UNDEFINED:
            message = '%s has no member named "%s"' % (repr(owner), name)
        else:
            message = '"%s" not defined' % name
        TemplateRuntimeError.__init__(self, message)