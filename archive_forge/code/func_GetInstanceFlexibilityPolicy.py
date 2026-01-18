from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def GetInstanceFlexibilityPolicy(dataproc, args, alpha):
    """Get instance flexibility policy.

  Args:
    dataproc: Dataproc object that contains client, messages, and resources
    args: arguments of the request
    alpha: checks if the release track is alpha

  Returns:
    InstanceFlexibilityPolicy of the secondary worker group.
  """
    if alpha and args.secondary_worker_standard_capacity_base is None:
        return None
    provisioning_model_mix = None
    instance_selection_list = []
    if alpha:
        provisioning_model_mix = dataproc.messages.ProvisioningModelMix(standardCapacityBase=args.secondary_worker_standard_capacity_base)
    else:
        instance_selection_list = GetInstanceSelectionList(dataproc, args)
    if provisioning_model_mix is None and (not instance_selection_list):
        return None
    instance_flexibility_policy = dataproc.messages.InstanceFlexibilityPolicy(instanceSelectionList=instance_selection_list, provisioningModelMix=provisioning_model_mix)
    return instance_flexibility_policy