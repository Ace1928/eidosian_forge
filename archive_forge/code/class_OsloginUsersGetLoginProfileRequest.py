from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersGetLoginProfileRequest(_messages.Message):
    """A OsloginUsersGetLoginProfileRequest object.

  Enums:
    ViewValueValuesEnum: The view configures whether to retrieve security keys
      information.

  Fields:
    name: Required. The unique ID for the user in format `users/{user}`.
    projectId: The project ID of the Google Cloud Platform project.
    systemId: A system ID for filtering the results of the request.
    view: The view configures whether to retrieve security keys information.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view configures whether to retrieve security keys information.

    Values:
      LOGIN_PROFILE_VIEW_UNSPECIFIED: The default login profile view. The API
        defaults to the BASIC view.
      BASIC: Includes POSIX and SSH key information.
      SECURITY_KEY: Include security key information for the user.
    """
        LOGIN_PROFILE_VIEW_UNSPECIFIED = 0
        BASIC = 1
        SECURITY_KEY = 2
    name = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2)
    systemId = _messages.StringField(3)
    view = _messages.EnumField('ViewValueValuesEnum', 4)