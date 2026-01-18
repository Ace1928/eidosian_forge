from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _labels_value(self, args: parser_extensions.Namespace) -> messages.VmwareNodeConfig.LabelsValue:
    """Constructs proto message LabelsValue."""
    node_labels = flags.Get(args, 'node_labels', {})
    additional_property_messages = []
    if not node_labels:
        return None
    for key, value in node_labels.items():
        additional_property_messages.append(messages.VmwareNodeConfig.LabelsValue.AdditionalProperty(key=key, value=value))
    labels_value_message = messages.VmwareNodeConfig.LabelsValue(additionalProperties=additional_property_messages)
    return labels_value_message