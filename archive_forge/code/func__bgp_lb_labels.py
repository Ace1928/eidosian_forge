from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bgp_lb_labels(self, args: parser_extensions.Namespace):
    """Constructs proto message LabelsValue."""
    node_labels = getattr(args, 'bgp_lb_load_balancer_node_labels', {})
    additional_property_messages = []
    if not node_labels:
        return None
    for key, value in node_labels.items():
        additional_property_messages.append(messages.BareMetalNodePoolConfig.LabelsValue.AdditionalProperty(key=key, value=value))
    labels_value_message = messages.BareMetalNodePoolConfig.LabelsValue(additionalProperties=additional_property_messages)
    return labels_value_message