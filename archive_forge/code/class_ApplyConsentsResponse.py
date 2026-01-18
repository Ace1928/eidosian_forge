from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyConsentsResponse(_messages.Message):
    """Response when all Consent resources in scope were processed and all
  affected resources were reindexed successfully. This structure is included
  in the response when the operation finishes successfully.

  Fields:
    affectedResources: The number of resources (including the Consent
      resources) that may have consensual access change.
    consentApplyFailure: If `validate_only = false` in ApplyConsentsRequest,
      this counter is the number of Consent resources that were failed to
      apply. Otherwise, it is the number of Consent resources that are not
      supported or invalid.
    consentApplySuccess: If `validate_only = false` in ApplyConsentsRequest,
      this counter is the number of Consent resources that were successfully
      applied. Otherwise, it is the number of Consent resources that are
      supported.
    failedResources: The number of resources (including the Consent resources)
      that ApplyConsents failed to re-index.
  """
    affectedResources = _messages.IntegerField(1)
    consentApplyFailure = _messages.IntegerField(2)
    consentApplySuccess = _messages.IntegerField(3)
    failedResources = _messages.IntegerField(4)