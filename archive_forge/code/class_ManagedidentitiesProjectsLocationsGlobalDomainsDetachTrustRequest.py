from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsDetachTrustRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsDetachTrustRequest
  object.

  Fields:
    detachTrustRequest: A DetachTrustRequest resource to be passed as the
      request body.
    name: Required. The resource domain name, project name, and location using
      the form: `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    detachTrustRequest = _messages.MessageField('DetachTrustRequest', 1)
    name = _messages.StringField(2, required=True)