from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from six.moves import range  # pylint: disable=redefined-builtin
def _GetNextPage(list_method, request, resource_field, page_token=None, limit=None):
    """Helper method to get the next set of paginated results.

  Args:
    list_method: The method that will execute the list request.
    request: The list request ready to be executed, possibly missing the page
        token.
    resource_field: The field name of the resources in the list results.
    page_token: The page token string to pass into the request, or None
        if no page token should be included.
    limit: Optional limit on how many resources to request.

  Returns:
    A tuple containing the list of results and the page token in the
    list response, or None if no page token was in the response.
  """
    if page_token:
        request.pageToken = page_token
    if limit:
        request.maxResults = limit
    try:
        response = list_method(request)
        return_token = response.nextPageToken
        results = response.get_assigned_value(resource_field) if response.get_assigned_value(resource_field) else []
        return (results, return_token)
    except apitools_exceptions.HttpError as error:
        raise api_exceptions.HttpException(error, HTTP_ERROR_FORMAT)