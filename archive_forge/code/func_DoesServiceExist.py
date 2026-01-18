from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def DoesServiceExist(service_name):
    """Check if a service resource exists.

  Args:
    service_name: name of the service to check if exists.

  Returns:
    Whether or not the service exists.
  """
    messages = GetMessagesModule()
    client = GetClientInstance()
    get_request = messages.ServicemanagementServicesGetRequest(serviceName=service_name)
    try:
        client.services.Get(get_request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError):
        return False
    else:
        return True