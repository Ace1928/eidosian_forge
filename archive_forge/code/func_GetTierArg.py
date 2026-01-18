from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetTierArg(messages):
    """Adds a --tier flag to the given parser.

  Args:
    messages: The messages module.

  Returns:
    the choice arg.
  """
    custom_mappings = {'STANDARD': ('standard', 'Standard Filestore instance, An alias for BASIC_HDD.\n            Use BASIC_HDD instead whenever possible.'), 'PREMIUM': ('premium', 'Premium Filestore instance, An alias for BASIC_SSD.\n                  Use BASIC_SSD instead whenever possible.'), 'BASIC_HDD': ('basic-hdd', 'Performant NFS storage system using HDD.'), 'BASIC_SSD': ('basic-ssd', 'Performant NFS storage system using SSD.'), 'ENTERPRISE': ('enterprise', 'Enterprise instance.\n            Use REGIONAL instead whenever possible.'), 'HIGH_SCALE_SSD': ('high-scale-ssd', 'High Scale SSD instance, an alias for ZONAL.\n            Use ZONAL instead whenever possible.'), 'ZONAL': ('zonal', 'Zonal instances offer NFS storage            system suitable for high performance computing application            requirements. It offers fast performance that scales            with capacity and allows you to grow and shrink            capacity.'), 'REGIONAL': ('regional', 'Regional instances offer the features          and availability needed for mission-critical workloads.')}
    tier_arg = arg_utils.ChoiceEnumMapper('--tier', messages.Instance.TierValueValuesEnum, help_str='The service tier for the Cloud Filestore instance.\n       For more details, see:\n       https://cloud.google.com/filestore/docs/instance-tiers ', custom_mappings=custom_mappings, default='BASIC_HDD')
    return tier_arg