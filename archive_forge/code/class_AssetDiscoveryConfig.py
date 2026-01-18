from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssetDiscoveryConfig(_messages.Message):
    """The configuration used for Asset Discovery runs.

  Enums:
    InclusionModeValueValuesEnum: The mode to use for filtering asset
      discovery.

  Fields:
    folderIds: The folder ids to use for filtering asset discovery. It
      consists of only digits, e.g., 756619654966.
    inclusionMode: The mode to use for filtering asset discovery.
    projectIds: The project ids to use for filtering asset discovery.
  """

    class InclusionModeValueValuesEnum(_messages.Enum):
        """The mode to use for filtering asset discovery.

    Values:
      INCLUSION_MODE_UNSPECIFIED: Unspecified. Setting the mode with this
        value will disable inclusion/exclusion filtering for Asset Discovery.
      INCLUDE_ONLY: Asset Discovery will capture only the resources within the
        projects specified. All other resources will be ignored.
      EXCLUDE: Asset Discovery will ignore all resources under the projects
        specified. All other resources will be retrieved.
    """
        INCLUSION_MODE_UNSPECIFIED = 0
        INCLUDE_ONLY = 1
        EXCLUDE = 2
    folderIds = _messages.StringField(1, repeated=True)
    inclusionMode = _messages.EnumField('InclusionModeValueValuesEnum', 2)
    projectIds = _messages.StringField(3, repeated=True)