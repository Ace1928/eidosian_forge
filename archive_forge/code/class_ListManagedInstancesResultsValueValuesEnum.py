from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListManagedInstancesResultsValueValuesEnum(_messages.Enum):
    """Pagination behavior of the listManagedInstances API method for this
    managed instance group.

    Values:
      PAGELESS: (Default) Pagination is disabled for the group's
        listManagedInstances API method. maxResults and pageToken query
        parameters are ignored and all instances are returned in a single
        response.
      PAGINATED: Pagination is enabled for the group's listManagedInstances
        API method. maxResults and pageToken query parameters are respected.
    """
    PAGELESS = 0
    PAGINATED = 1