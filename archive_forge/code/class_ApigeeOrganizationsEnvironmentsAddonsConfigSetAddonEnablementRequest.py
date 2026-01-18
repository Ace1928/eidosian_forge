from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsAddonsConfigSetAddonEnablementRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsAddonsConfigSetAddonEnablementRequest
  object.

  Fields:
    googleCloudApigeeV1SetAddonEnablementRequest: A
      GoogleCloudApigeeV1SetAddonEnablementRequest resource to be passed as
      the request body.
    name: Required. Name of the add-ons config. Must be in the format of
      `/organizations/{org}/environments/{env}/addonsConfig`
  """
    googleCloudApigeeV1SetAddonEnablementRequest = _messages.MessageField('GoogleCloudApigeeV1SetAddonEnablementRequest', 1)
    name = _messages.StringField(2, required=True)