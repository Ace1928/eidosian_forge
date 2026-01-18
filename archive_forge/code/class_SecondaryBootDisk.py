from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecondaryBootDisk(_messages.Message):
    """SecondaryBootDisk represents a persistent disk attached to a node with
  special configurations based on its mode.

  Enums:
    ModeValueValuesEnum: Disk mode (container image cache, etc.)

  Fields:
    diskImage: Fully-qualified resource ID for an existing disk image.
    mode: Disk mode (container image cache, etc.)
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Disk mode (container image cache, etc.)

    Values:
      MODE_UNSPECIFIED: MODE_UNSPECIFIED is when mode is not set.
      CONTAINER_IMAGE_CACHE: CONTAINER_IMAGE_CACHE is for using the secondary
        boot disk as a container image cache.
    """
        MODE_UNSPECIFIED = 0
        CONTAINER_IMAGE_CACHE = 1
    diskImage = _messages.StringField(1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)