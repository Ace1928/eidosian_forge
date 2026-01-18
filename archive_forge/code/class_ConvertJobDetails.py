from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConvertJobDetails(_messages.Message):
    """Details regarding a Convert background job.

  Fields:
    filter: Output only. AIP-160 based filter used to specify the entities to
      convert
  """
    filter = _messages.StringField(1)