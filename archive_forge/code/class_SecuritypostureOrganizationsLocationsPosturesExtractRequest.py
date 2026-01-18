from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPosturesExtractRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPosturesExtractRequest object.

  Fields:
    extractPostureRequest: A ExtractPostureRequest resource to be passed as
      the request body.
    parent: Required. The parent resource name. The format of this value is as
      follows: `organizations/{organization}/locations/{location}`
  """
    extractPostureRequest = _messages.MessageField('ExtractPostureRequest', 1)
    parent = _messages.StringField(2, required=True)