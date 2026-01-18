from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActivateConsentRequest(_messages.Message):
    """Activates the latest revision of the specified Consent by committing a
  new revision with `state` updated to `ACTIVE`. If the latest revision of the
  given Consent is in the `ACTIVE` state, no new revision is committed. A
  FAILED_PRECONDITION error occurs if the latest revision of the given consent
  is in the `REJECTED` or `REVOKED` state.

  Fields:
    consentArtifact: Required. The resource name of the Consent artifact that
      contains documentation of the user's consent, of the form `projects/{pro
      ject_id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{co
      nsent_store_id}/consentArtifacts/{consent_artifact_id}`. If the draft
      Consent had a Consent artifact, this Consent artifact overwrites it.
    expireTime: Timestamp in UTC of when this Consent is considered expired.
    ttl: The time to live for this Consent from when it is marked as active.
  """
    consentArtifact = _messages.StringField(1)
    expireTime = _messages.StringField(2)
    ttl = _messages.StringField(3)