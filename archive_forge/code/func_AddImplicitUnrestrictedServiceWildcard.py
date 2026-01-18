from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def AddImplicitUnrestrictedServiceWildcard(ref, args, req):
    """Add wildcard for unrestricted services to message if type is regular.

  Args:
    ref: resources.Resource, the (unused) resource
    args: argparse namespace, the parse arguments
    req: AccesscontextmanagerAccessPoliciesAccessZonesCreateRequest

  Returns:
    The modified request.
  """
    del ref, args
    m = util.GetMessages(version='v1beta')
    if req.servicePerimeter.perimeterType == m.ServicePerimeter.PerimeterTypeValueValuesEnum.PERIMETER_TYPE_REGULAR:
        service_perimeter_config = req.servicePerimeter.status
        if not service_perimeter_config:
            service_perimeter_config = m.ServicePerimeterConfig
        service_perimeter_config.unrestrictedServices = ['*']
        req.servicePerimeter.status = service_perimeter_config
    return req