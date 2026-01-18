from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedConfig(_messages.Message):
    """ManagedConfig is used for enforcing set of cluster configurations that
  are conforming to strandards.

  Enums:
    TypeValueValuesEnum: The type of standard configurations to enforce for
      cluster.

  Fields:
    type: The type of standard configurations to enforce for cluster.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of standard configurations to enforce for cluster.

    Values:
      TYPE_UNSPECIFIED: Default value.
      DISABLED: ManagedConfig is disabled.
      AUTOFLEET: Use ManagedConfig that is conforming to Autofleet
        requirements.
    """
        TYPE_UNSPECIFIED = 0
        DISABLED = 1
        AUTOFLEET = 2
    type = _messages.EnumField('TypeValueValuesEnum', 1)