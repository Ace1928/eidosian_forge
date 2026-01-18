from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaBucketId(_messages.Message):
    """A quota bucket is an instance of a quota limit.

  Fields:
    containerId: A Quota limit is defined at container level ORGANIZATION,
      PROJECT. The container of a quota bucket for a quota limit is identified
      by organization id, or project id respectively.
    region: If a quota limit is defined on PER_REGION dimension, then each
      region will have its own quota bucket. This field is non-empty only if
      the quota limit is defined on PER_REGION dimension.
    zone: If a quota limit is defined on PER_ZONE dimension, then each zone
      will have its own quota bucket. This field is non-empty only if the
      quota limit is defined on PER_ZONE dimension.
  """
    containerId = _messages.StringField(1)
    region = _messages.StringField(2)
    zone = _messages.StringField(3)