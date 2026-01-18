from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Metadata(_messages.Message):
    """Metadata represents the JSON encoded generated customer metadata.

  Fields:
    metadata: JSON encoded Google-generated Customer Metadata for a given
      resource/project.
  """
    metadata = _messages.StringField(1)