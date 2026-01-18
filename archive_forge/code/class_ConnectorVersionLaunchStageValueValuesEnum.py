from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorVersionLaunchStageValueValuesEnum(_messages.Enum):
    """Output only. Flag to mark the version indicating the launch stage.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: LAUNCH_STAGE_UNSPECIFIED.
      PREVIEW: PREVIEW.
      GA: GA.
      DEPRECATED: DEPRECATED.
      PRIVATE_PREVIEW: PRIVATE_PREVIEW.
    """
    LAUNCH_STAGE_UNSPECIFIED = 0
    PREVIEW = 1
    GA = 2
    DEPRECATED = 3
    PRIVATE_PREVIEW = 4