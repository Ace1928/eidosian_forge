from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersExclusionsPatchRequest(_messages.Message):
    """A LoggingFoldersExclusionsPatchRequest object.

  Fields:
    logExclusion: A LogExclusion resource to be passed as the request body.
    name: Required. The resource name of the exclusion to update:
      "projects/[PROJECT_ID]/exclusions/[EXCLUSION_ID]"
      "organizations/[ORGANIZATION_ID]/exclusions/[EXCLUSION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/exclusions/[EXCLUSION_ID]"
      "folders/[FOLDER_ID]/exclusions/[EXCLUSION_ID]" For
      example:"projects/my-project/exclusions/my-exclusion"
    updateMask: Required. A non-empty list of fields to change in the existing
      exclusion. New values for the fields are taken from the corresponding
      fields in the LogExclusion included in this request. Fields not
      mentioned in update_mask are not changed and are ignored in the
      request.For example, to change the filter and description of an
      exclusion, specify an update_mask of "filter,description".
  """
    logExclusion = _messages.MessageField('LogExclusion', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)