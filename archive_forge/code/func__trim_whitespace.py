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
def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str]=_ws_re) -> str:
    return _ws_re.sub(' ', string.strip())