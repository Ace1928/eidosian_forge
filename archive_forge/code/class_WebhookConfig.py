from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebhookConfig(_messages.Message):
    """WebhookConfig describes the configuration of a trigger that creates a
  build whenever a webhook is sent to a trigger's webhook URL.

  Enums:
    StateValueValuesEnum: Potential issues with the underlying Pub/Sub
      subscription configuration. Only populated on get requests.

  Fields:
    secret: Required. Resource name for the secret required as a URL
      parameter.
    state: Potential issues with the underlying Pub/Sub subscription
      configuration. Only populated on get requests.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Potential issues with the underlying Pub/Sub subscription
    configuration. Only populated on get requests.

    Values:
      STATE_UNSPECIFIED: The webhook auth configuration not been checked.
      OK: The auth configuration is properly setup.
      SECRET_DELETED: The secret provided in auth_method has been deleted.
    """
        STATE_UNSPECIFIED = 0
        OK = 1
        SECRET_DELETED = 2
    secret = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)