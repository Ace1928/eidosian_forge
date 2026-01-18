from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagersListManagedInstancesResponse(_messages.Message):
    """A InstanceGroupManagersListManagedInstancesResponse object.

  Fields:
    managedInstances: [Output Only] The list of instances in the managed
      instance group.
    nextPageToken: [Output Only] This token allows you to get the next page of
      results for list requests. If the number of results is larger than
      maxResults, use the nextPageToken as a value for the query parameter
      pageToken in the next list request. Subsequent list requests will have
      their own nextPageToken to continue paging through the results.
  """
    managedInstances = _messages.MessageField('ManagedInstance', 1, repeated=True)
    nextPageToken = _messages.StringField(2)