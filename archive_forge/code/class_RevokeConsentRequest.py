from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevokeConsentRequest(_messages.Message):
    """Revokes the latest revision of the specified Consent by committing a new
  revision with `state` updated to `REVOKED`. If the latest revision of the
  given Consent is in the `REVOKED` state, no new revision is committed.
  """