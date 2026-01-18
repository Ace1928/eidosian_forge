from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicLevel(_messages.Message):
    """`BasicLevel` is an `AccessLevel` using a set of recommended features.

  Enums:
    CombiningFunctionValueValuesEnum: How the `conditions` list should be
      combined to determine if a request is granted this `AccessLevel`. If AND
      is used, each `Condition` in `conditions` must be satisfied for the
      `AccessLevel` to be applied. If OR is used, at least one `Condition` in
      `conditions` must be satisfied for the `AccessLevel` to be applied.
      Default behavior is AND.

  Fields:
    combiningFunction: How the `conditions` list should be combined to
      determine if a request is granted this `AccessLevel`. If AND is used,
      each `Condition` in `conditions` must be satisfied for the `AccessLevel`
      to be applied. If OR is used, at least one `Condition` in `conditions`
      must be satisfied for the `AccessLevel` to be applied. Default behavior
      is AND.
    conditions: Required. A list of requirements for the `AccessLevel` to be
      granted.
  """

    class CombiningFunctionValueValuesEnum(_messages.Enum):
        """How the `conditions` list should be combined to determine if a request
    is granted this `AccessLevel`. If AND is used, each `Condition` in
    `conditions` must be satisfied for the `AccessLevel` to be applied. If OR
    is used, at least one `Condition` in `conditions` must be satisfied for
    the `AccessLevel` to be applied. Default behavior is AND.

    Values:
      AND: All `Conditions` must be true for the `BasicLevel` to be true.
      OR: If at least one `Condition` is true, then the `BasicLevel` is true.
    """
        AND = 0
        OR = 1
    combiningFunction = _messages.EnumField('CombiningFunctionValueValuesEnum', 1)
    conditions = _messages.MessageField('Condition', 2, repeated=True)