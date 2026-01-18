from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SandboxConfig(_messages.Message):
    """SandboxConfig contains configurations of the sandbox to use for the
  node.

  Enums:
    TypeValueValuesEnum: Type of the sandbox to use for the node.

  Fields:
    type: Type of the sandbox to use for the node.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the sandbox to use for the node.

    Values:
      UNSPECIFIED: Default value. This should not be used.
      GVISOR: Run sandbox using gvisor.
    """
        UNSPECIFIED = 0
        GVISOR = 1
    type = _messages.EnumField('TypeValueValuesEnum', 1)