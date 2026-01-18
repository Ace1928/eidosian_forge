from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeidentifiedStoreDestination(_messages.Message):
    """Contains configuration for streaming de-identified FHIR export.

  Fields:
    config: The configuration to use when de-identifying resources that are
      added to this store.
    store: The full resource name of a Cloud Healthcare FHIR store, for
      example, `projects/{project_id}/locations/{location_id}/datasets/{datase
      t_id}/fhirStores/{fhir_store_id}`.
  """
    config = _messages.MessageField('DeidentifyConfig', 1)
    store = _messages.StringField(2)