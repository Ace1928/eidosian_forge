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
def AddAuthenticatorSecurityGroupFlags(parser, hidden=False):
    """Adds --security-group to parser."""
    help_text = 'The name of the RBAC security group for use with Google security groups\nin Kubernetes RBAC\n(https://kubernetes.io/docs/reference/access-authn-authz/rbac/).\n\nTo include group membership as part of the claims issued by Google\nduring authentication, a group must be designated as a security group by\nincluding it as a direct member of this group.\n\nIf unspecified, no groups will be returned for use with RBAC.'
    parser.add_argument('--security-group', help=help_text, default=None, hidden=hidden)