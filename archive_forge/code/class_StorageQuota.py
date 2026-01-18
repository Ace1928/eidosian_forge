from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageQuota(_messages.Message):
    """A storage provisioning quota .

  Fields:
    availableGib: Storage size (GiB).
    gcpService: The gcp service of the provisioning quota.
    name: Output only. The name of the provisioning quota.
  """
    availableGib = _messages.IntegerField(1)
    gcpService = _messages.StringField(2)
    name = _messages.StringField(3)