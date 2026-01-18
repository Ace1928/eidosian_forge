from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddResourceManagerTagsCreate(parser, for_node_pool=False):
    """Adds a --resource-manager-tags to the given parser on create."""
    if for_node_pool:
        help_text = 'Applies the specified comma-separated resource manager tags that has the\nGCE_FIREWALL purpose to all nodes in the new node pool.\n\nExamples:\n\n  $ {command} example-node-pool --resource-manager-tags=tagKeys/1234=tagValues/2345\n  $ {command} example-node-pool --resource-manager-tags=my-project/key1=value1\n  $ {command} example-node-pool --resource-manager-tags=12345/key1=value1,23456/key2=value2\n  $ {command} example-node-pool --resource-manager-tags=\n\nAll nodes, including nodes that are resized or re-created, will have the\nspecified tags on the corresponding Instance object in the Compute Engine API.\nYou can reference these tags in network firewall policy rules. For instructions,\nsee https://cloud.google.com/firewall/docs/use-tags-for-firewalls.\n'
    else:
        help_text = 'Applies the specified comma-separated resource manager tags that has the\nGCE_FIREWALL purpose to all nodes in the new default node pool(s) of a new cluster.\n\nExamples:\n\n  $ {command} example-cluster --resource-manager-tags=tagKeys/1234=tagValues/2345\n  $ {command} example-cluster --resource-manager-tags=my-project/key1=value1\n  $ {command} example-cluster --resource-manager-tags=12345/key1=value1,23456/key2=value2\n  $ {command} example-cluster --resource-manager-tags=\n\nAll nodes, including nodes that are resized or re-created, will have the\nspecified tags on the corresponding Instance object in the Compute Engine API.\nYou can reference these tags in network firewall policy rules. For instructions,\nsee https://cloud.google.com/firewall/docs/use-tags-for-firewalls.\n'
    AddResourceManagerTagsFlag(parser, help_text)