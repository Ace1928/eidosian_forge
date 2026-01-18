from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudFunctionsV2OperationMetadata(_messages.Message):
    """Represents the metadata of the long-running operation.

  Enums:
    OperationTypeValueValuesEnum: The operation type.

  Messages:
    RequestResourceValue: The original request that started the operation.

  Fields:
    apiVersion: API version used to start the operation.
    cancelRequested: Identifies whether the user has requested cancellation of
      the operation. Operations that have successfully been cancelled have
      google.longrunning.Operation.error value with a google.rpc.Status.code
      of 1, corresponding to `Code.CANCELLED`.
    createTime: The time the operation was created.
    endTime: The time the operation finished running.
    operationType: The operation type.
    requestResource: The original request that started the operation.
    sourceToken: An identifier for Firebase function sources. Disclaimer: This
      field is only supported for Firebase function deployments.
    stages: Mechanism for reporting in-progress stages
    statusDetail: Human-readable status of the operation, if any.
    target: Server-defined resource path for the target of the operation.
    verb: Name of the verb executed by the operation.
  """

    class OperationTypeValueValuesEnum(_messages.Enum):
        """The operation type.

    Values:
      OPERATIONTYPE_UNSPECIFIED: Unspecified
      CREATE_FUNCTION: CreateFunction
      UPDATE_FUNCTION: UpdateFunction
      DELETE_FUNCTION: DeleteFunction
      REDIRECT_FUNCTION_UPGRADE_TRAFFIC: RedirectFunctionUpgradeTraffic
      ROLLBACK_FUNCTION_UPGRADE_TRAFFIC: RollbackFunctionUpgradeTraffic
      SETUP_FUNCTION_UPGRADE_CONFIG: SetupFunctionUpgradeConfig
      ABORT_FUNCTION_UPGRADE: AbortFunctionUpgrade
      COMMIT_FUNCTION_UPGRADE: CommitFunctionUpgrade
    """
        OPERATIONTYPE_UNSPECIFIED = 0
        CREATE_FUNCTION = 1
        UPDATE_FUNCTION = 2
        DELETE_FUNCTION = 3
        REDIRECT_FUNCTION_UPGRADE_TRAFFIC = 4
        ROLLBACK_FUNCTION_UPGRADE_TRAFFIC = 5
        SETUP_FUNCTION_UPGRADE_CONFIG = 6
        ABORT_FUNCTION_UPGRADE = 7
        COMMIT_FUNCTION_UPGRADE = 8

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestResourceValue(_messages.Message):
        """The original request that started the operation.

    Messages:
      AdditionalProperty: An additional property for a RequestResourceValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestResourceValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    apiVersion = _messages.StringField(1)
    cancelRequested = _messages.BooleanField(2)
    createTime = _messages.StringField(3)
    endTime = _messages.StringField(4)
    operationType = _messages.EnumField('OperationTypeValueValuesEnum', 5)
    requestResource = _messages.MessageField('RequestResourceValue', 6)
    sourceToken = _messages.StringField(7)
    stages = _messages.MessageField('GoogleCloudFunctionsV2Stage', 8, repeated=True)
    statusDetail = _messages.StringField(9)
    target = _messages.StringField(10)
    verb = _messages.StringField(11)