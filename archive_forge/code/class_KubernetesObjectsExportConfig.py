from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesObjectsExportConfig(_messages.Message):
    """KubernetesObjectsExportConfig is configuration which enables export of
  kubernetes resource changes to specified targets.

  Fields:
    kubernetesObjectsChangesTarget: Target to which objects changes should be
      sent. Currently the only supported value here is CLOUD_LOGGING.
    kubernetesObjectsSnapshotsTarget: Target to which objects snapshots should
      be sent. Currently the only supported value here is CLOUD_LOGGING.
  """
    kubernetesObjectsChangesTarget = _messages.StringField(1)
    kubernetesObjectsSnapshotsTarget = _messages.StringField(2)