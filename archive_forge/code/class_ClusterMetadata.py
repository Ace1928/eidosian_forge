from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterMetadata(_messages.Message):
    """Information about the GKE cluster from which this Backup was created.

  Messages:
    BackupCrdVersionsValue: Output only. A list of the Backup for GKE CRD
      versions found in the cluster.

  Fields:
    anthosVersion: Output only. Anthos version
    backupCrdVersions: Output only. A list of the Backup for GKE CRD versions
      found in the cluster.
    cluster: Output only. The source cluster from which this Backup was
      created. Valid formats: - `projects/*/locations/*/clusters/*` -
      `projects/*/zones/*/clusters/*` This is inherited from the parent
      BackupPlan's cluster field.
    gkeVersion: Output only. GKE version
    k8sVersion: Output only. The Kubernetes server version of the source
      cluster.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BackupCrdVersionsValue(_messages.Message):
        """Output only. A list of the Backup for GKE CRD versions found in the
    cluster.

    Messages:
      AdditionalProperty: An additional property for a BackupCrdVersionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        BackupCrdVersionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BackupCrdVersionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    anthosVersion = _messages.StringField(1)
    backupCrdVersions = _messages.MessageField('BackupCrdVersionsValue', 2)
    cluster = _messages.StringField(3)
    gkeVersion = _messages.StringField(4)
    k8sVersion = _messages.StringField(5)