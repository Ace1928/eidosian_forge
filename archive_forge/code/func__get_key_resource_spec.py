from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _get_key_resource_spec(api_version):
    """Return the resource specification for a key."""
    if api_version == 'v2':
        return concepts.ResourceSpec('apikeys.projects.locations.keys', api_version=api_version, resource_name='key', disable_auto_completers=True, keysId=_key_attribute_config(), locationsId=_location_attribute_config(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)
    else:
        return concepts.ResourceSpec('apikeys.projects.keys', api_version=api_version, resource_name='key', disable_auto_completers=True, keysId=_key_attribute_config(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)