from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServiceusageV1Service(_messages.Message):
    """A service that is available for use by the consumer.

  Enums:
    StateValueValuesEnum: Whether or not the service has been enabled for use
      by the consumer.

  Fields:
    config: The service configuration of the available service. Some fields
      may be filtered out of the configuration in responses to the
      `ListServices` method. These fields are present only in responses to the
      `GetService` method.
    name: The resource name of the consumer and service.  A valid name would
      be: - projects/123/services/serviceusage.googleapis.com
    parent: The resource name of the consumer.  A valid name would be: -
      projects/123
    state: Whether or not the service has been enabled for use by the
      consumer.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Whether or not the service has been enabled for use by the consumer.

    Values:
      STATE_UNSPECIFIED: The default value, which indicates that the enabled
        state of the service is unspecified or not meaningful. Currently, all
        consumers other than projects (such as folders and organizations) are
        always in this state.
      DISABLED: The service cannot be used by this consumer. It has either
        been explicitly disabled, or has never been enabled.
      ENABLED: The service has been explicitly enabled for use by this
        consumer.
    """
        STATE_UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2
    config = _messages.MessageField('GoogleApiServiceusageV1ServiceConfig', 1)
    name = _messages.StringField(2)
    parent = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)