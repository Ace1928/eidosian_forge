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
@pass_context
def _gettext_alias(__context: Context, *args: t.Any, **kwargs: t.Any) -> t.Union[t.Any, Undefined]:
    return __context.call(__context.resolve('gettext'), *args, **kwargs)