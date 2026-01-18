from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsGetXpnResources(_messages.Message):
    """A ProjectsGetXpnResources object.

  Fields:
    kind: [Output Only] Type of resource. Always
      compute#projectsGetXpnResources for lists of service resources (a.k.a
      service projects)
    nextPageToken: [Output Only] This token allows you to get the next page of
      results for list requests. If the number of results is larger than
      maxResults, use the nextPageToken as a value for the query parameter
      pageToken in the next list request. Subsequent list requests will have
      their own nextPageToken to continue paging through the results.
    resources: Service resources (a.k.a service projects) attached to this
      project as their shared VPC host.
  """
    kind = _messages.StringField(1, default='compute#projectsGetXpnResources')
    nextPageToken = _messages.StringField(2)
    resources = _messages.MessageField('XpnResourceId', 3, repeated=True)