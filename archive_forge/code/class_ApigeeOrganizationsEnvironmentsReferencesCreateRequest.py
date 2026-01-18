from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsReferencesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsReferencesCreateRequest object.

  Fields:
    googleCloudApigeeV1Reference: A GoogleCloudApigeeV1Reference resource to
      be passed as the request body.
    parent: Required. The parent environment name under which the Reference
      will be created. Must be of the form
      `organizations/{org}/environments/{env}`.
  """
    googleCloudApigeeV1Reference = _messages.MessageField('GoogleCloudApigeeV1Reference', 1)
    parent = _messages.StringField(2, required=True)