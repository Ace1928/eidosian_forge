from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RejectConsentRequest(_messages.Message):
    """Rejects the latest revision of the specified Consent by committing a new
  revision with `state` updated to `REJECTED`. If the latest revision of the
  given Consent is in the `REJECTED` state, no new revision is committed.

  Fields:
    consentArtifact: Optional. The resource name of the Consent artifact that
      contains documentation of the user's rejection of the draft Consent, of
      the form `projects/{project_id}/locations/{location_id}/datasets/{datase
      t_id}/consentStores/{consent_store_id}/consentArtifacts/{consent_artifac
      t_id}`. If the draft Consent had a Consent artifact, this Consent
      artifact overwrites it.
  """
    consentArtifact = _messages.StringField(1)