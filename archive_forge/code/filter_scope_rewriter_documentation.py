from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
import six
Rewrites restrictions for keys in self._keys.

    Args:
      key: The dotted resource name.
      op: The operator name.
      operand: The operand string value.
      key_type: The type of key, None if not known.

    Returns:
      A specific set of operands or None.
    