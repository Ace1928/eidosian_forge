from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelCallToActionOpenNotebooks(_messages.Message):
    """Open notebooks.

  Fields:
    notebooks: Required. Regional resource references to notebooks.
  """
    notebooks = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToActionRegionalResourceReferences', 1, repeated=True)