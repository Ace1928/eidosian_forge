from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationCluster(_messages.Message):
    """A workstation cluster resource in the Cloud Workstations API. Defines a
  group of workstations in a particular region and the VPC network they're
  attached to.

  Messages:
    AnnotationsValue: Optional. Client-specified annotations.
    LabelsValue: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation cluster and that are also propagated
      to the underlying Compute Engine resources.

  Fields:
    annotations: Optional. Client-specified annotations.
    conditions: Output only. Status conditions describing the workstation
      cluster's current state.
    controlPlaneIp: Output only. The private IP address of the control plane
      for this workstation cluster. Workstation VMs need access to this IP
      address to work with the service, so make sure that your firewall rules
      allow egress from the workstation VMs to this address.
    createTime: Output only. Time when this workstation cluster was created.
    degraded: Output only. Whether this workstation cluster is in degraded
      mode, in which case it may require user action to restore full
      functionality. Details can be found in conditions.
    deleteTime: Output only. Time when this workstation cluster was soft-
      deleted.
    displayName: Optional. Human-readable name for this workstation cluster.
    domainConfig: Optional. Configuration options for a custom domain.
    etag: Optional. Checksum computed by the server. May be sent on update and
      delete requests to make sure that the client has an up-to-date value
      before proceeding.
    labels: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation cluster and that are also propagated
      to the underlying Compute Engine resources.
    name: Identifier. Full name of this workstation cluster.
    network: Immutable. Name of the Compute Engine network in which instances
      associated with this workstation cluster will be created.
    privateClusterConfig: Optional. Configuration for private workstation
      cluster.
    reconciling: Output only. Indicates whether this workstation cluster is
      currently being updated to match its intended state.
    subnetwork: Immutable. Name of the Compute Engine subnetwork in which
      instances associated with this workstation cluster will be created. Must
      be part of the subnetwork specified for this workstation cluster.
    uid: Output only. A system-assigned unique identifier for this workstation
      cluster.
    updateTime: Output only. Time when this workstation cluster was most
      recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Client-specified annotations.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. [Labels](https://cloud.google.com/workstations/docs/label-
    resources) that are applied to the workstation cluster and that are also
    propagated to the underlying Compute Engine resources.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    conditions = _messages.MessageField('Status', 2, repeated=True)
    controlPlaneIp = _messages.StringField(3)
    createTime = _messages.StringField(4)
    degraded = _messages.BooleanField(5)
    deleteTime = _messages.StringField(6)
    displayName = _messages.StringField(7)
    domainConfig = _messages.MessageField('DomainConfig', 8)
    etag = _messages.StringField(9)
    labels = _messages.MessageField('LabelsValue', 10)
    name = _messages.StringField(11)
    network = _messages.StringField(12)
    privateClusterConfig = _messages.MessageField('PrivateClusterConfig', 13)
    reconciling = _messages.BooleanField(14)
    subnetwork = _messages.StringField(15)
    uid = _messages.StringField(16)
    updateTime = _messages.StringField(17)