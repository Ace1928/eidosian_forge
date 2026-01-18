import logging
import re
from oslo_policy import _checks
@reducer('or_expr', 'or', 'check')
def _extend_or_expr(self, or_expr, _or, check):
    """Extend an 'or_expr' by adding one more check."""
    return [('or_expr', or_expr.add_check(check))]