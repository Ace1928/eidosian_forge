import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def _render_error(template, context, error):
    if template.error_handler:
        result = template.error_handler(context, error)
        if not result:
            tp, value, tb = sys.exc_info()
            if value and tb:
                raise value.with_traceback(tb)
            else:
                raise error
    else:
        error_template = exceptions.html_error_template()
        if context._outputting_as_unicode:
            context._buffer_stack[:] = [util.FastEncodingBuffer()]
        else:
            context._buffer_stack[:] = [util.FastEncodingBuffer(error_template.output_encoding, error_template.encoding_errors)]
        context._set_with_template(error_template)
        error_template.render_context(context, error=error)