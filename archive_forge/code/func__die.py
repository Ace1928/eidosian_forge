from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
def _die(self, *args, **kwargs):
    """Raise an `UndefinedError`."""
    __traceback_hide__ = True
    raise UndefinedError(self._name, self._owner)