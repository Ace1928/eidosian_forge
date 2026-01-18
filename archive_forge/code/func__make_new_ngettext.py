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
def _make_new_ngettext(func: t.Callable[[str, str, int], str]) -> t.Callable[..., str]:

    @pass_context
    def ngettext(__context: Context, __singular: str, __plural: str, __num: int, **variables: t.Any) -> str:
        variables.setdefault('num', __num)
        rv = __context.call(func, __singular, __plural, __num)
        if __context.eval_ctx.autoescape:
            rv = Markup(rv)
        return rv % variables
    return ngettext