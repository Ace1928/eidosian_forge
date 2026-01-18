from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeedJobDetails(_messages.Message):
    """Details regarding a Seed background job.

  Fields:
    connectionProfile: Output only. The connection profile which was used for
      the seed job.
  """
    connectionProfile = _messages.StringField(1)