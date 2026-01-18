import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
class DebugExtension(Extension):
    """A ``{% debug %}`` tag that dumps the available variables,
    filters, and tests.

    .. code-block:: html+jinja

        <pre>{% debug %}</pre>

    .. code-block:: text

        {'context': {'cycler': <class 'jinja2.utils.Cycler'>,
                     ...,
                     'namespace': <class 'jinja2.utils.Namespace'>},
         'filters': ['abs', 'attr', 'batch', 'capitalize', 'center', 'count', 'd',
                     ..., 'urlencode', 'urlize', 'wordcount', 'wordwrap', 'xmlattr'],
         'tests': ['!=', '<', '<=', '==', '>', '>=', 'callable', 'defined',
                   ..., 'odd', 'sameas', 'sequence', 'string', 'undefined', 'upper']}

    .. versionadded:: 2.11.0
    """
    tags = {'debug'}

    def parse(self, parser: 'Parser') -> nodes.Output:
        lineno = parser.stream.expect('name:debug').lineno
        context = nodes.ContextReference()
        result = self.call_method('_render', [context], lineno=lineno)
        return nodes.Output([result], lineno=lineno)

    def _render(self, context: Context) -> str:
        result = {'context': context.get_all(), 'filters': sorted(self.environment.filters.keys()), 'tests': sorted(self.environment.tests.keys())}
        return pprint.pformat(result, depth=3, compact=True)