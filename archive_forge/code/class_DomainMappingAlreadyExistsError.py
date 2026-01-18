from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class DomainMappingAlreadyExistsError(DomainMappingCreationError):
    """Domain mapping already exists in another project, GCP service, or region.

  This indicates a succesfully created DomainMapping resource but with the
  domain it intends to map being unavailable because it's already in use.
  Not to be confused with a 409 error indicating a DomainMapping resource with
  this same name (the domain name) already exists in this region.
  """