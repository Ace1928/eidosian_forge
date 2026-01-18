from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
def _RewriteTimeTerm(key, op, operand):
    """Rewrites <createTime op operand>."""
    if op not in ['<', '<=', '=', ':', '>=', '>']:
        return None
    try:
        dt = times.ParseDateTime(operand)
    except ValueError as e:
        raise ValueError('{operand}: date-time value expected for {key}: {error}'.format(operand=operand, key=key, error=six.text_type(e)))
    if op == ':':
        op = '='
    return '{key} {op} "{dt}"'.format(key=key, op=op, dt=times.FormatDateTime(dt, tzinfo=times.UTC))