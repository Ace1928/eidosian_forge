from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresImportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresImportRequest object.

  Fields:
    importMessagesRequest: A ImportMessagesRequest resource to be passed as
      the request body.
    name: Required. The name of the target HL7v2 store, in the format `project
      s/{project_id}/locations/{location_id}/datasets/{dataset_id}/hl7v2Stores
      /{hl7v2_store_id}`
  """
    importMessagesRequest = _messages.MessageField('ImportMessagesRequest', 1)
    name = _messages.StringField(2, required=True)