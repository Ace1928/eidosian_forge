from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Hl7V2StoreMetrics(_messages.Message):
    """List of metrics for a given HL7v2 store.

  Fields:
    metrics: List of HL7v2 store metrics by message type.
    name: The resource name of the HL7v2 store to get metrics for, in the
      format `projects/{project_id}/datasets/{dataset_id}/hl7V2Stores/{hl7v2_s
      tore_id}`.
  """
    metrics = _messages.MessageField('Hl7V2StoreMetric', 1, repeated=True)
    name = _messages.StringField(2)