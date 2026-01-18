from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _AddInstanceSelection(messages, instance_selections, instance_selection_name, machine_types, rank):
    """Adds instance selection to instance selections list."""
    for instance_selection in instance_selections:
        if instance_selection.key == instance_selection_name:
            if instance_selection.value is not None:
                raise InvalidArgumentError('Attempt to add multiple instance selections with the same name.')
            instance_selections.remove(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.AdditionalProperty(key=instance_selection_name, value=None))
    if rank is not None:
        instance_selections.append(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.AdditionalProperty(key=instance_selection_name, value=messages.InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection(rank=rank, machineTypes=machine_types)))
    else:
        instance_selections.append(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.AdditionalProperty(key=instance_selection_name, value=messages.InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection(machineTypes=machine_types)))