from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAcceleratorTypeMapper(version):
    """Get a mapper for accelerator type to enum value."""
    if version == constants.BETA_VERSION:
        return arg_utils.ChoiceEnumMapper('generic-accelerator', apis.GetMessagesModule(constants.AI_PLATFORM_API_NAME, constants.AI_PLATFORM_API_VERSION[version]).GoogleCloudAiplatformV1beta1MachineSpec.AcceleratorTypeValueValuesEnum, help_str='The available types of accelerators.', include_filter=lambda x: x.startswith('NVIDIA'), required=False)
    return arg_utils.ChoiceEnumMapper('generic-accelerator', apis.GetMessagesModule(constants.AI_PLATFORM_API_NAME, constants.AI_PLATFORM_API_VERSION[version]).GoogleCloudAiplatformV1MachineSpec.AcceleratorTypeValueValuesEnum, help_str='The available types of accelerators.', include_filter=lambda x: x.startswith('NVIDIA'), required=False)