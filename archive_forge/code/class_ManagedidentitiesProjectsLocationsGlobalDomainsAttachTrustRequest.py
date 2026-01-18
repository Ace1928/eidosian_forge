from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsAttachTrustRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsAttachTrustRequest
  object.

  Fields:
    attachTrustRequest: A AttachTrustRequest resource to be passed as the
      request body.
    name: Required. The resource domain name, project name and location using
      the form: `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    attachTrustRequest = _messages.MessageField('AttachTrustRequest', 1)
    name = _messages.StringField(2, required=True)