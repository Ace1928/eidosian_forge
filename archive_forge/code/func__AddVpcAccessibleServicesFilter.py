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
def _AddVpcAccessibleServicesFilter(args, req, version):
    """Add the particular service filter message based on specified args."""
    service_restriction_config = None
    allowed_services = None
    enable_restriction = None
    restriction_modified = False
    service_perimeter_config = req.servicePerimeter.status
    if not service_perimeter_config:
        service_perimeter_config = util.GetMessages(version=version).ServicePerimeterConfig
    if args.IsSpecified('vpc_allowed_services'):
        allowed_services = getattr(args, 'vpc_allowed_services')
        restriction_modified = True
    if args.IsSpecified('enable_vpc_accessible_services'):
        enable_restriction = getattr(args, 'enable_vpc_accessible_services')
        restriction_modified = True
    if restriction_modified:
        service_restriction_config = getattr(service_perimeter_config, 'vpcAccessibleServices')
        if not service_restriction_config:
            service_restriction_config = getattr(util.GetMessages(version=version), 'VpcAccessibleServices')
        service_restriction_config.allowedServices = allowed_services
        service_restriction_config.enableRestriction = enable_restriction
    setattr(service_perimeter_config, 'vpcAccessibleServices', service_restriction_config)
    req.servicePerimeter.status = service_perimeter_config
    return req