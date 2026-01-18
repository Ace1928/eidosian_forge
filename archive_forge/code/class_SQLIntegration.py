from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SQLIntegration(_messages.Message):
    """Represents the SQL instance integrated with AD.

  Enums:
    StateValueValuesEnum: Output only. The current state of the managed OU.

  Fields:
    createTime: Output only. The time the instance was created.
    name: The unique name of the sql integration in the form of `projects/{pro
      ject_id}/locations/global/domains/{domain_name}/sqlIntegrations/{sql_int
      egration}`
    sqlInstance: The full resource name of an integrated sql instance
    state: Output only. The current state of the managed OU.
    updateTime: Output only. Last update time for this SQL instance.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the managed OU.

    Values:
      STATE_UNSPECIFIED: Not Set
      CREATING: The sqlIntegration is being created.
      DELETING: The sqlIntegration is being deleted.
      READY: The sqlIntegration is ready.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        DELETING = 2
        READY = 3
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    sqlInstance = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    updateTime = _messages.StringField(5)