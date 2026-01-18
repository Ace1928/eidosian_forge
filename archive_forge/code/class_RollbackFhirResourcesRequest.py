from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackFhirResourcesRequest(_messages.Message):
    """Request to roll back resources.

  Enums:
    ChangeTypeValueValuesEnum: Optional. CREATE/UPDATE/DELETE/ALL for
      reverting all txns of a certain type.

  Fields:
    changeType: Optional. CREATE/UPDATE/DELETE/ALL for reverting all txns of a
      certain type.
    excludeRollbacks: Optional. Specifies whether to exclude earlier
      rollbacks.
    filteringFields: Optional. Tag represents fields that HDE needs to
      identify resources that will be reverted. Parameters for filtering
      resources
    force: Optional. When enabled, changes will be reverted without explicit
      confirmation
    inputGcsObject: Optional. Cloud Storage object containing list of
      {resourceType}/{resourceId} lines, identifying resources to be reverted
    resultGcsBucket: Required. Bucket to deposit result
    rollbackTime: Required. Time point to rollback to.
    type: Optional. If specified, revert only resources of these types
  """

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """Optional. CREATE/UPDATE/DELETE/ALL for reverting all txns of a certain
    type.

    Values:
      CHANGE_TYPE_UNSPECIFIED: When unspecified, revert all transactions
      ALL: All transactions
      CREATE: Revert only CREATE transactions
      UPDATE: Revert only Update transactions
      DELETE: Revert only Delete transactions
    """
        CHANGE_TYPE_UNSPECIFIED = 0
        ALL = 1
        CREATE = 2
        UPDATE = 3
        DELETE = 4
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 1)
    excludeRollbacks = _messages.BooleanField(2)
    filteringFields = _messages.MessageField('RollbackFhirResourceFilteringFields', 3)
    force = _messages.BooleanField(4)
    inputGcsObject = _messages.StringField(5)
    resultGcsBucket = _messages.StringField(6)
    rollbackTime = _messages.StringField(7)
    type = _messages.StringField(8, repeated=True)