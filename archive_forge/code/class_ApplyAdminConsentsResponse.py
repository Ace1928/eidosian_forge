from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyAdminConsentsResponse(_messages.Message):
    """Response when all admin Consent resources in scope were processed and
  all affected resources were reindexed successfully. This structure will be
  included in the response when the operation finishes successfully.

  Fields:
    affectedResources: The number of resources (including the Consent
      resources) that may have consent access change.
    consentApplySuccess: If `validate_only=false` in
      ApplyAdminConsentsRequest, this counter contains the number of Consent
      resources that were successfully applied. Otherwise, it is the number of
      Consent resources that are supported.
    failedResources: The number of resources (including the Consent resources)
      that ApplyAdminConsents failed to re-index.
  """
    affectedResources = _messages.IntegerField(1)
    consentApplySuccess = _messages.IntegerField(2)
    failedResources = _messages.IntegerField(3)