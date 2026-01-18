from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsGetRequest(_messages.Message):
    """A AppengineAppsServicesVersionsGetRequest object.

  Enums:
    IncludeExtraDataValueValuesEnum: Optional. Options to include extra data
    ViewValueValuesEnum: Controls the set of fields returned in the Get
      response.

  Fields:
    includeExtraData: Optional. Options to include extra data
    name: Name of the resource requested. Example:
      apps/myapp/services/default/versions/v1.
    view: Controls the set of fields returned in the Get response.
  """

    class IncludeExtraDataValueValuesEnum(_messages.Enum):
        """Optional. Options to include extra data

    Values:
      INCLUDE_EXTRA_DATA_UNSPECIFIED: Unspecified: No extra data will be
        returned
      INCLUDE_EXTRA_DATA_NONE: Do not return any extra data
      INCLUDE_GOOGLE_GENERATED_METADATA: Return GGCM associated with the
        resources
    """
        INCLUDE_EXTRA_DATA_UNSPECIFIED = 0
        INCLUDE_EXTRA_DATA_NONE = 1
        INCLUDE_GOOGLE_GENERATED_METADATA = 2

    class ViewValueValuesEnum(_messages.Enum):
        """Controls the set of fields returned in the Get response.

    Values:
      BASIC: Basic version information including scaling and inbound services,
        but not detailed deployment information.
      FULL: The information from BASIC, plus detailed information about the
        deployment. This format is required when creating resources, but is
        not returned in Get or List by default.
    """
        BASIC = 0
        FULL = 1
    includeExtraData = _messages.EnumField('IncludeExtraDataValueValuesEnum', 1)
    name = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)