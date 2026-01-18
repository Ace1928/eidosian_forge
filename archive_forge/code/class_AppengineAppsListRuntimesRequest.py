from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsListRuntimesRequest(_messages.Message):
    """A AppengineAppsListRuntimesRequest object.

  Enums:
    EnvironmentValueValuesEnum: Optional. The environment of the Application.

  Fields:
    environment: Optional. The environment of the Application.
    parent: Required. Name of the parent Application resource. Example:
      apps/myapp.
  """

    class EnvironmentValueValuesEnum(_messages.Enum):
        """Optional. The environment of the Application.

    Values:
      ENVIRONMENT_UNSPECIFIED: Default value.
      STANDARD: App Engine Standard.
      FLEXIBLE: App Engine Flexible
    """
        ENVIRONMENT_UNSPECIFIED = 0
        STANDARD = 1
        FLEXIBLE = 2
    environment = _messages.EnumField('EnvironmentValueValuesEnum', 1)
    parent = _messages.StringField(2, required=True)