from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroup(_messages.Message):
    """Dataproc Node Group. The Dataproc NodeGroup resource is not related to
  the Dataproc NodeGroupAffinity resource.

  Enums:
    RolesValueListEntryValuesEnum:

  Messages:
    LabelsValue: Optional. Node group labels. Label keys must consist of from
      1 to 63 characters and conform to RFC 1035
      (https://www.ietf.org/rfc/rfc1035.txt). Label values can be empty. If
      specified, they must consist of from 1 to 63 characters and conform to
      RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). The node group must
      have no more than 32 labelsn.

  Fields:
    labels: Optional. Node group labels. Label keys must consist of from 1 to
      63 characters and conform to RFC 1035
      (https://www.ietf.org/rfc/rfc1035.txt). Label values can be empty. If
      specified, they must consist of from 1 to 63 characters and conform to
      RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt). The node group must
      have no more than 32 labelsn.
    name: The Node group resource name (https://aip.dev/122).
    nodeGroupConfig: Optional. The node group instance group configuration.
    roles: Required. Node group roles.
  """

    class RolesValueListEntryValuesEnum(_messages.Enum):
        """RolesValueListEntryValuesEnum enum type.

    Values:
      ROLE_UNSPECIFIED: Required unspecified role.
      DRIVER: Job drivers run on the node pool.
      MASTER: Master nodes.
      PRIMARY_WORKER: Primary workers.
      SECONDARY_WORKER: Secondary workers.
    """
        ROLE_UNSPECIFIED = 0
        DRIVER = 1
        MASTER = 2
        PRIMARY_WORKER = 3
        SECONDARY_WORKER = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Node group labels. Label keys must consist of from 1 to 63
    characters and conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).
    Label values can be empty. If specified, they must consist of from 1 to 63
    characters and conform to RFC 1035 (https://www.ietf.org/rfc/rfc1035.txt).
    The node group must have no more than 32 labelsn.

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
    labels = _messages.MessageField('LabelsValue', 1)
    name = _messages.StringField(2)
    nodeGroupConfig = _messages.MessageField('InstanceGroupConfig', 3)
    roles = _messages.EnumField('RolesValueListEntryValuesEnum', 4, repeated=True)