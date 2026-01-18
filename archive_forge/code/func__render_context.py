import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _render_context(tmpl, callable_, context, *args, **kwargs):
    import mako.template as template
    if not isinstance(tmpl, template.DefTemplate):
        inherit, lclcontext = _populate_self_namespace(context, tmpl)
        _exec_template(inherit, lclcontext, args=args, kwargs=kwargs)
    else:
        inherit, lclcontext = _populate_self_namespace(context, tmpl.parent)
        _exec_template(callable_, context, args=args, kwargs=kwargs)