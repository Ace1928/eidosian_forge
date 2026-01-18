from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigSyncVersion(_messages.Message):
    """Specific versioning information pertaining to ConfigSync's Pods

  Fields:
    admissionWebhook: Version of the deployed admission_webhook pod
    gitSync: Version of the deployed git-sync pod
    importer: Version of the deployed importer pod
    monitor: Version of the deployed monitor pod
    reconcilerManager: Version of the deployed reconciler-manager pod
    rootReconciler: Version of the deployed reconciler container in root-
      reconciler pod
    syncer: Version of the deployed syncer pod
  """
    admissionWebhook = _messages.StringField(1)
    gitSync = _messages.StringField(2)
    importer = _messages.StringField(3)
    monitor = _messages.StringField(4)
    reconcilerManager = _messages.StringField(5)
    rootReconciler = _messages.StringField(6)
    syncer = _messages.StringField(7)