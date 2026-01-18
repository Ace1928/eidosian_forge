from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsCreateRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsCreateRequest object.

  Fields:
    domain: A Domain resource to be passed as the request body.
    domainName: Required. The fully qualified domain name. e.g.
      mydomain.myorganization.com, with the following restrictions: * Must
      contain only lowercase letters, numbers, periods and hyphens. * Must
      start with a letter. * Must contain between 2-64 characters. * Must end
      with a number or a letter. * Must not start with period. * First segment
      length (mydomain for example above) shouldn't exceed 15 chars. * The
      last segment cannot be fully numeric. * Must be unique within the
      customer project.
    parent: Required. The resource project name and location using the form:
      `projects/{project_id}/locations/global`
  """
    domain = _messages.MessageField('Domain', 1)
    domainName = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)