from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class Code(object):
    """Abstract base class for the `Expression` and `Suite` classes."""
    __slots__ = ['source', 'code', 'ast', '_globals']

    def __init__(self, source, filename=None, lineno=-1, lookup='strict', xform=None):
        """Create the code object, either from a string, or from an AST node.
        
        :param source: either a string containing the source code, or an AST
                       node
        :param filename: the (preferably absolute) name of the file containing
                         the code
        :param lineno: the number of the line on which the code was found
        :param lookup: the lookup class that defines how variables are looked
                       up in the context; can be either "strict" (the default),
                       "lenient", or a custom lookup class
        :param xform: the AST transformer that should be applied to the code;
                      if `None`, the appropriate transformation is chosen
                      depending on the mode
        """
        if isinstance(source, six.string_types):
            self.source = source
            node = _parse(source, mode=self.mode)
        else:
            assert isinstance(source, _ast.AST), 'Expected string or AST node, but got %r' % source
            self.source = '?'
            if self.mode == 'eval':
                node = _ast.Expression()
                node.body = source
            else:
                node = _ast.Module()
                node.body = [source]
        self.ast = node
        self.code = _compile(node, self.source, mode=self.mode, filename=filename, lineno=lineno, xform=xform)
        if lookup is None:
            lookup = LenientLookup
        elif isinstance(lookup, six.string_types):
            lookup = {'lenient': LenientLookup, 'strict': StrictLookup}[lookup]
        self._globals = lookup.globals

    def __getstate__(self):
        if hasattr(self._globals, '__self__'):
            lookup_fn = self._globals.__self__
        else:
            lookup_fn = self._globals.im_self
        state = {'source': self.source, 'ast': self.ast, 'lookup': lookup_fn}
        state['code'] = get_code_params(self.code)
        return state

    def __setstate__(self, state):
        self.source = state['source']
        self.ast = state['ast']
        self.code = CodeType(0, *state['code'])
        self._globals = state['lookup'].globals

    def __eq__(self, other):
        return type(other) == type(self) and self.code == other.code

    def __hash__(self):
        return hash(self.code)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.source)