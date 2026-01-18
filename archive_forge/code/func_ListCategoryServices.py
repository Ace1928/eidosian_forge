import collections
import copy
import enum
import sys
from typing import List
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
def ListCategoryServices(resource, category, page_size=200, limit=sys.maxsize):
    """Make API call to list category services .

  Args:
    resource: resource to get list for. format-"projects/100", "folders/101" or
      "organizations/102".
    category: category to get list for. format-"catgeory/<category>".
    page_size: The page size to list.default=200
    limit: The max number of services to display.

  Raises:
    exceptions.ListCategoryServicespermissionDeniedException: when listing the
    services the parent category includes.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The services the parent category includes.
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageCategoriesCategoryServicesListRequest(parent='{}/{}'.format(resource, category))
    try:
        return list_pager.YieldFromList(_Lister(client.categories_categoryServices), request, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field='services')
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.ListCategoryServicespermissionDeniedException)