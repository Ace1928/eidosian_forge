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
class _TranslationsBasic(te.Protocol):

    def gettext(self, message: str) -> str:
        ...

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        pass