from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SquashModeValueValuesEnum(_messages.Enum):
    """Either NO_ROOT_SQUASH, for allowing root access on the exported
    directory, or ROOT_SQUASH, for not allowing root access. The default is
    NO_ROOT_SQUASH.

    Values:
      SQUASH_MODE_UNSPECIFIED: SquashMode not set.
      NO_ROOT_SQUASH: The Root user has root access to the file share
        (default).
      ROOT_SQUASH: The Root user has squashed access to the anonymous uid/gid.
    """
    SQUASH_MODE_UNSPECIFIED = 0
    NO_ROOT_SQUASH = 1
    ROOT_SQUASH = 2