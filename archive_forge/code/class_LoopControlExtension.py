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
class LoopControlExtension(Extension):
    """Adds break and continue to the template engine."""
    tags = {'break', 'continue'}

    def parse(self, parser: 'Parser') -> t.Union[nodes.Break, nodes.Continue]:
        token = next(parser.stream)
        if token.value == 'break':
            return nodes.Break(lineno=token.lineno)
        return nodes.Continue(lineno=token.lineno)