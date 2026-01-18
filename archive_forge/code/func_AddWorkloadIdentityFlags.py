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
def AddWorkloadIdentityFlags(parser, use_identity_provider=False):
    """Adds Workload Identity flags to the parser."""
    parser.add_argument('--workload-pool', default=None, help='Enable Workload Identity on the cluster.\n\nWhen enabled, Kubernetes service accounts will be able to act as Cloud IAM\nService Accounts, through the provided workload pool.\n\nCurrently, the only accepted workload pool is the workload pool of\nthe Cloud project containing the cluster, `PROJECT_ID.svc.id.goog`.\n\nFor more information on Workload Identity, see\n\n            https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity\n  ', required=False, type=arg_parsers.RegexpValidator('^[a-z][-a-z0-9]{4,}[a-z0-9]\\.(svc|hub)\\.id\\.goog$', "Must be in format of '[PROJECT_ID].svc.id.goog'"))
    if use_identity_provider:
        parser.add_argument('--identity-provider', default=None, help='  Enable 3P identity provider on the cluster.\n    ')