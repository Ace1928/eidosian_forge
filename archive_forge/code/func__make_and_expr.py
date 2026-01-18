import logging
import re
from oslo_policy import _checks
@reducer('check', 'and', 'check')
def _make_and_expr(self, check1, _and, check2):
    """Create an 'and_expr'.

        Join two checks by the 'and' operator.
        """
    return [('and_expr', _checks.AndCheck([check1, check2]))]