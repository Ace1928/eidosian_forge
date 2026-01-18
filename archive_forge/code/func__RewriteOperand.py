from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.core.resource import resource_expr_rewrite
import six
def _RewriteOperand(self, key, operand):
    if isinstance(operand, list):
        return [self._RewriteOperand(key, operand_item) for operand_item in operand]
    return self._KEY_OPERAND_MAPPING.get(key, {}).get(operand, operand)