from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
class OperationsBackend(resource_expr_rewrite.Backend):
    """Limit filter expressions to those supported by the Genomics backend."""
    _FORMAT = '{key} {op} {operand}'
    _QUOTED_FORMAT = '{key} {op} "{operand}"'
    _TERMS = {'^done$': _FORMAT, '^error.code$': _FORMAT, '^metadata.labels\\.(.*)': _QUOTED_FORMAT, '^metadata.events$': _QUOTED_FORMAT}
    _CREATE_TIME_TERMS = ['^metadata.create_time$', '^metadata.createTime$']

    def RewriteTerm(self, key, op, operand, key_type):
        """Limit <key op operand> terms to expressions supported by the backend."""
        for regex in self._CREATE_TIME_TERMS:
            if re.match(regex, key):
                return _RewriteTimeTerm(key, op, operand)
        for regex, fmt in six.iteritems(self._TERMS):
            if re.match(regex, key):
                return fmt.format(key=key, op=op, operand=operand)
        return None