from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
def FetchLogs(log_filter=None, order_by='DESC', limit=None, parent=None, resource_names=None):
    """Fetches log entries.

  This method uses Cloud Logging V2 api.
  https://cloud.google.com/logging/docs/api/introduction_v2

  Entries are sorted on the timestamp field, and afterwards filter is applied.
  If limit is passed, returns only up to that many matching entries.

  If neither log_filter nor log_ids are passed, no filtering is done.

  FetchLogs will query the combined resource set from "parent" and
  "resource_names".

  Args:
    log_filter: filter expression used in the request.
    order_by: the sort order, either DESC or ASC.
    limit: how many entries to return.
    parent: the name of the log's parent resource, e.g. "projects/foo" or
      "organizations/123" or "folders/123". Defaults to the current project if
      no `resource_names` are provided.
    resource_names: if present, resource names to query.

  Returns:
    A generator that returns matching log entries.
    Callers are responsible for handling any http exceptions.
  """
    resource_names = resource_names or []
    for name in resource_names:
        _AssertValidResource('resource_names', name)
    if parent:
        _AssertValidResource('parent', parent)
        resource_names.append(parent)
    elif not resource_names:
        resource_names.append('projects/%s' % properties.VALUES.core.project.Get(required=True))
    page_size = min(limit or 1000, 1000)
    if order_by.upper() == 'DESC':
        order_by = 'timestamp desc'
    else:
        order_by = 'timestamp asc'
    client = util.GetClient()
    request = client.MESSAGES_MODULE.ListLogEntriesRequest(resourceNames=resource_names, filter=log_filter, orderBy=order_by)
    return list_pager.YieldFromList(client.entries, request, field='entries', limit=limit, batch_size=page_size, batch_size_attribute='pageSize')