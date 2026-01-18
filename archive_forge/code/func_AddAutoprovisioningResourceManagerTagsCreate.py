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
def AddAutoprovisioningResourceManagerTagsCreate(parser):
    """Adds a --autoprovisioning-resource-manager-tags to the given parser on create."""
    help_text = 'Applies the specified comma-separated resource manager tags that has the\nGCE_FIREWALL purpose to all nodes in the new Autopilot cluster or\nall auto-provisioned nodes in the new Standard cluster.\n\nExamples:\n\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=tagKeys/1234=tagValues/2345\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=my-project/key1=value1\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=12345/key1=value1,23456/key2=value2\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=\n\nAll nodes in an Autopilot cluster or all auto-provisioned nodes in a Standard\ncluster, including nodes that are resized or re-created, will have the specified\ntags on the corresponding Instance object in the Compute Engine API. You can\nreference these tags in network firewall policy rules. For instructions, see\nhttps://cloud.google.com/firewall/docs/use-tags-for-firewalls.\n'
    AddAutoprovisioningResourceManagerTagsFlag(parser, help_text)