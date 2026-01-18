from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesProjectSettingsGetRequest(_messages.Message):
    """A ServicemanagementServicesProjectSettingsGetRequest object.

  Enums:
    ViewValueValuesEnum: Request only the fields for the specified view.

  Fields:
    consumerProjectId: The project ID of the consumer.
    expand: Fields to expand in any results.  By default, the following fields
      are not present in the result: - `operations` - `quota_usage`
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
    view: Request only the fields for the specified view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Request only the fields for the specified view.

    Values:
      PROJECT_SETTINGS_VIEW_UNSPECIFIED: <no description>
      CONSUMER_VIEW: <no description>
      PRODUCER_VIEW: <no description>
      ALL: <no description>
    """
        PROJECT_SETTINGS_VIEW_UNSPECIFIED = 0
        CONSUMER_VIEW = 1
        PRODUCER_VIEW = 2
        ALL = 3
    consumerProjectId = _messages.StringField(1, required=True)
    expand = _messages.StringField(2)
    serviceName = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)