import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _exec_template(callable_, context, args=None, kwargs=None):
    """execute a rendering callable given the callable, a
    Context, and optional explicit arguments

    the contextual Template will be located if it exists, and
    the error handling options specified on that Template will
    be interpreted here.
    """
    template = context._with_template
    if template is not None and (template.format_exceptions or template.error_handler):
        try:
            callable_(context, *args, **kwargs)
        except Exception:
            _render_error(template, context, compat.exception_as())
        except:
            e = sys.exc_info()[0]
            _render_error(template, context, e)
    else:
        callable_(context, *args, **kwargs)