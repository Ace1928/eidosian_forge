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
def AddAccessLevelsBase(ref, args, req, version=None):
    """Hook to add access levels to request."""
    if args.IsSpecified('access_levels'):
        access_levels = []
        for access_level in args.access_levels:
            if access_level.startswith('accessPolicies'):
                access_levels.append(access_level)
            else:
                level_ref = resources.REGISTRY.Create('accesscontextmanager.accessPolicies.accessLevels', accessLevelsId=access_level, **ref.Parent().AsDict())
                access_levels.append(level_ref.RelativeName())
        service_perimeter_config = req.servicePerimeter.status
        if not service_perimeter_config:
            service_perimeter_config = util.GetMessages(version=version).ServicePerimeterConfig
        service_perimeter_config.accessLevels = access_levels
        req.servicePerimeter.status = service_perimeter_config
    return req