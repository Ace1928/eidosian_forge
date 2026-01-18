from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Tier(_messages.Message):
    """A Google Cloud SQL service tier resource.

  Fields:
    DiskQuota: The maximum disk size of this tier in bytes.
    RAM: The maximum RAM usage of this tier in bytes.
    edition: Edition can be STANDARD or ENTERPRISE.
    kind: This is always `sql#tier`.
    region: The applicable regions for this tier.
    tier: An identifier for the machine type, for example, `db-custom-1-3840`.
      For related information, see [Pricing](/sql/pricing).
  """
    DiskQuota = _messages.IntegerField(1)
    RAM = _messages.IntegerField(2)
    edition = _messages.StringField(3)
    kind = _messages.StringField(4)
    region = _messages.StringField(5, repeated=True)
    tier = _messages.StringField(6)