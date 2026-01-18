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
def _make_new_gettext(func: t.Callable[[str], str]) -> t.Callable[..., str]:

    @pass_context
    def gettext(__context: Context, __string: str, **variables: t.Any) -> str:
        rv = __context.call(func, __string)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        return rv % variables
    return gettext