from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
def _RewriteTimes(self, key, op, operand):
    """Rewrites <*Time op operand>."""
    try:
        dt = times.ParseDateTime(operand)
    except ValueError as e:
        raise ValueError('{operand}: date-time value expected for {key}: {error}'.format(operand=operand, key=key, error=six.text_type(e)))
    dt_string = times.FormatDateTime(dt, '%Y-%m-%dT%H:%M:%S.%3f%Ez', times.UTC)
    return '{key}{op}{dt_string}'.format(key=key, op=op, dt_string=self.Quote(dt_string, always=True))