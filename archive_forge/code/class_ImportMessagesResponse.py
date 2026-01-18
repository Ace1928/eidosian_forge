from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportMessagesResponse(_messages.Message):
    """Final response of importing messages. This structure is included in the
  response to describe the detailed outcome. It is only included when the
  operation finishes successfully.

  Fields:
    hl7v2Store: The name of the target HL7v2 store, in the format `projects/{p
      roject_id}/locations/{location_id}/datasets/{dataset_id}/hl7v2Stores/{hl
      7v2_store_id}`
    inputSize: The total number of resources included in the source data.
  """
    hl7v2Store = _messages.StringField(1)
    inputSize = _messages.IntegerField(2)