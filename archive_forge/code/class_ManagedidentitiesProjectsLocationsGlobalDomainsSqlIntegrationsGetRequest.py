from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsGetRequest(_messages.Message):
    """A
  ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsGetRequest
  object.

  Fields:
    name: Required. SQLIntegration resource name using the form: `projects/{pr
      oject_id}/locations/global/domains/{domain}/sqlIntegrations/{name}`
  """
    name = _messages.StringField(1, required=True)