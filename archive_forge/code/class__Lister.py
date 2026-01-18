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
class _Lister:

    def __init__(self, service_usage):
        self.service_usage = service_usage

    @http_retry.RetryOnHttpStatus(_TOO_MANY_REQUESTS)
    def List(self, request, global_params=None):
        return self.service_usage.List(request, global_params=global_params)