from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsValidateTrustRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsValidateTrustRequest
  object.

  Fields:
    name: Required. The resource domain name, project name, and location using
      the form: `projects/{project_id}/locations/global/domains/{domain_name}`
    validateTrustRequest: A ValidateTrustRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    validateTrustRequest = _messages.MessageField('ValidateTrustRequest', 2)