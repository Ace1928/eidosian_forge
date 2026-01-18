from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
def _get_template_frame_from_source(source):
    if not source:
        return None
    origin, (start, end) = source
    filename = getattr(origin, 'loadname', None)
    if filename is None:
        filename = '<django template>'
    template_source = origin.reload()
    lineno = None
    upto = 0
    pre_context = []
    post_context = []
    context_line = None
    for num, next in enumerate(_linebreak_iter(template_source)):
        line = template_source[upto:next]
        if start >= upto and end <= next:
            lineno = num
            context_line = line
        elif lineno is None:
            pre_context.append(line)
        else:
            post_context.append(line)
        upto = next
    if context_line is None or lineno is None:
        return None
    return {'filename': filename, 'lineno': lineno, 'pre_context': pre_context[-5:], 'post_context': post_context[:5], 'context_line': context_line}