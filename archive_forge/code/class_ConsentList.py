from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentList(_messages.Message):
    """List of resource names of Consent resources.

  Fields:
    consents: The resource names of the Consents to evaluate against, of the
      form `projects/{project_id}/locations/{location_id}/datasets/{dataset_id
      }/consentStores/{consent_store_id}/consents/{consent_id}`.
  """
    consents = _messages.StringField(1, repeated=True)