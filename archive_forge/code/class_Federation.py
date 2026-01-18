from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Federation(_messages.Message):
    """Represents a federation of multiple backend metastores.

  Enums:
    StateValueValuesEnum: Output only. The current state of the federation.

  Messages:
    BackendMetastoresValue: A map from BackendMetastore rank to
      BackendMetastores from which the federation service serves metadata at
      query time. The map key represents the order in which BackendMetastores
      should be evaluated to resolve database names at query time and should
      be greater than or equal to zero. A BackendMetastore with a lower number
      will be evaluated before a BackendMetastore with a higher number.
    LabelsValue: User-defined labels for the metastore federation.

  Fields:
    backendMetastores: A map from BackendMetastore rank to BackendMetastores
      from which the federation service serves metadata at query time. The map
      key represents the order in which BackendMetastores should be evaluated
      to resolve database names at query time and should be greater than or
      equal to zero. A BackendMetastore with a lower number will be evaluated
      before a BackendMetastore with a higher number.
    createTime: Output only. The time when the metastore federation was
      created.
    endpointUri: Output only. The federation endpoint.
    labels: User-defined labels for the metastore federation.
    name: Immutable. The relative resource name of the federation, of the
      form: projects/{project_number}/locations/{location_id}/federations/{fed
      eration_id}`.
    state: Output only. The current state of the federation.
    stateMessage: Output only. Additional information about the current state
      of the metastore federation, if available.
    uid: Output only. The globally unique resource identifier of the metastore
      federation.
    updateTime: Output only. The time when the metastore federation was last
      updated.
    version: Immutable. The Apache Hive metastore version of the federation.
      All backend metastore versions must be compatible with the federation
      version.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the federation.

    Values:
      STATE_UNSPECIFIED: The state of the metastore federation is unknown.
      CREATING: The metastore federation is in the process of being created.
      ACTIVE: The metastore federation is running and ready to serve queries.
      UPDATING: The metastore federation is being updated. It remains usable
        but cannot accept additional update requests or be deleted at this
        time.
      DELETING: The metastore federation is undergoing deletion. It cannot be
        used.
      ERROR: The metastore federation has encountered an error and cannot be
        used. The metastore federation should be deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        UPDATING = 3
        DELETING = 4
        ERROR = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BackendMetastoresValue(_messages.Message):
        """A map from BackendMetastore rank to BackendMetastores from which the
    federation service serves metadata at query time. The map key represents
    the order in which BackendMetastores should be evaluated to resolve
    database names at query time and should be greater than or equal to zero.
    A BackendMetastore with a lower number will be evaluated before a
    BackendMetastore with a higher number.

    Messages:
      AdditionalProperty: An additional property for a BackendMetastoresValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        BackendMetastoresValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BackendMetastoresValue object.

      Fields:
        key: Name of the additional property.
        value: A BackendMetastore attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('BackendMetastore', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels for the metastore federation.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    backendMetastores = _messages.MessageField('BackendMetastoresValue', 1)
    createTime = _messages.StringField(2)
    endpointUri = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stateMessage = _messages.StringField(7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    version = _messages.StringField(10)