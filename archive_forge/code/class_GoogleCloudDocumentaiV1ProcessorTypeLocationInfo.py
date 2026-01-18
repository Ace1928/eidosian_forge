from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessorTypeLocationInfo(_messages.Message):
    """The location information about where the processor is available.

  Fields:
    locationId: The location ID. For supported locations, refer to [regional
      and multi-regional support](/document-ai/docs/regions).
  """
    locationId = _messages.StringField(1)