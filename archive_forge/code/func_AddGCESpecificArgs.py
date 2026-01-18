from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddGCESpecificArgs(parser):
    """Add GCE specific arguments to parser."""
    gce_arg_group = parser.add_argument_group(help='Parameters for Google Compute Engine instance identity tokens.')
    gce_arg_group.add_argument('--token-format', choices=['standard', 'full'], default='standard', help='Specify whether or not the project and instance details are included in the identity token payload. This flag only applies to Google Compute Engine instance identity tokens. See https://cloud.google.com/compute/docs/instances/verifying-instance-identity#token_format for more details on token format.')
    gce_arg_group.add_argument('--include-license', action='store_true', help='Specify whether or not license codes for images associated with this instance are included in the identity token payload. Default is False. This flag does not have effect unless `--token-format=full`.')