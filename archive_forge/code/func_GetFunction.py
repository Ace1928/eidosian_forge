from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import util as util_v1
from googlecloudsdk.api_lib.functions.v2 import types
from googlecloudsdk.api_lib.functions.v2 import util
from googlecloudsdk.core import properties
import six
@util_v1.CatchHTTPErrorRaiseHTTPException
def GetFunction(self, name: str, raise_if_not_found: bool=False) -> Optional[types.Function]:
    """Gets the function with the given name or None if not found.

    Args:
      name: GCFv2 function resource relative name.
      raise_if_not_found: If set, raises NOT_FOUND http errors instead of
        returning None.

    Returns:
      cloudfunctions_v2_messages.Function, the fetched GCFv2 function or None.
    """
    try:
        return self.client.projects_locations_functions.Get(self.messages.CloudfunctionsProjectsLocationsFunctionsGetRequest(name=name))
    except apitools_exceptions.HttpError as error:
        if raise_if_not_found or error.status_code != six.moves.http_client.NOT_FOUND:
            raise
        return None