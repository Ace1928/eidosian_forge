from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
def _get_template_frame_from_debug(debug):
    if debug is None:
        return None
    lineno = debug['line']
    filename = debug['name']
    if filename is None:
        filename = '<django template>'
    pre_context = []
    post_context = []
    context_line = None
    for i, line in debug['source_lines']:
        if i < lineno:
            pre_context.append(line)
        elif i > lineno:
            post_context.append(line)
        else:
            context_line = line
    return {'filename': filename, 'lineno': lineno, 'pre_context': pre_context[-5:], 'post_context': post_context[:5], 'context_line': context_line, 'in_app': True}