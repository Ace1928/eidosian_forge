from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreatePatchInstanceFilter(messages, filter_all, filter_group_labels, filter_zones, filter_names, filter_name_prefixes):
    """Creates a PatchInstanceFilter message from its components."""
    group_labels = []
    for group_label in filter_group_labels:
        pairs = []
        for key, value in group_label.items():
            pairs.append(messages.PatchInstanceFilterGroupLabel.LabelsValue.AdditionalProperty(key=key, value=value))
        group_labels.append(messages.PatchInstanceFilterGroupLabel(labels=messages.PatchInstanceFilterGroupLabel.LabelsValue(additionalProperties=pairs)))
    return messages.PatchInstanceFilter(all=filter_all, groupLabels=group_labels, zones=filter_zones, instances=filter_names, instanceNamePrefixes=filter_name_prefixes)