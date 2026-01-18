from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsDeleteRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsDeleteRequest object.

  Fields:
    name: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    name = _messages.StringField(1, required=True)