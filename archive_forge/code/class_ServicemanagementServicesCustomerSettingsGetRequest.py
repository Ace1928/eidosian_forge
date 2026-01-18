from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesCustomerSettingsGetRequest(_messages.Message):
    """A ServicemanagementServicesCustomerSettingsGetRequest object.

  Enums:
    ViewValueValuesEnum: Request only fields for the specified view.

  Fields:
    customerId: ID for the customer. See the comment for
      `CustomerSettings.customer_id` field of message for its format. This
      field is required.
    expand: Fields to expand in any results.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`. This
      field is required.
    view: Request only fields for the specified view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Request only fields for the specified view.

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
    customerId = _messages.StringField(1, required=True)
    expand = _messages.StringField(2)
    serviceName = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)