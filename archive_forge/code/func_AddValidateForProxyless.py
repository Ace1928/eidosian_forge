from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddValidateForProxyless(parser):
    """Adds the validate_for_proxyless argument."""
    parser.add_argument('--validate-for-proxyless', action='store_true', default=False, help='      If specified, configuration in the associated urlMap and the\n      BackendServices is checked to allow only the features that are supported\n      in the latest release of gRPC.\n\n      If unspecified, no such configuration checks are performed. This may cause\n      unexpected behavior in gRPC applications if unsupported features are\n      configured.\n      ')