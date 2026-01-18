from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.kms import resource_args as kms_args
from googlecloudsdk.command_lib.privateca import completers as privateca_completers
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def ValidateResourceLocation(resource_ref, arg_name, version='v1'):
    """Raises an exception if the given resource is in an unsupported location."""
    supported_locations = locations.GetSupportedLocations(version=version)
    if resource_ref.locationsId not in supported_locations:
        raise exceptions.InvalidArgumentException(arg_name, 'Resource is in an unsupported location. Supported locations are: {}.'.format(', '.join(sorted(supported_locations))))