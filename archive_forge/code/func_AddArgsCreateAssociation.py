from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsCreateAssociation(parser):
    """Adds the arguments of association creation."""
    parser.add_argument('--firewall-policy', required=True, help='Security policy ID of the association.')
    parser.add_argument('--organization', help='ID of the organization in which the firewall policy is to be associated. Must be set if FIREWALL_POLICY is short name.')
    parser.add_argument('--folder', help='ID of the folder with which the association is created.')
    parser.add_argument('--replace-association-on-target', action='store_true', default=False, required=False, help='By default, if you attempt to insert an association to an organization or folder resource that is already associated with a firewall policy the method will fail. If this is set, the existing  association will be deleted at the same time that the new association is created.')
    parser.add_argument('--name', help='Name to identify this association. If unspecified, the name will be set to "organization-{ORGANIZATION_ID}" or "folder-{FOLDER_ID}".')