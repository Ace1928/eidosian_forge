from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddConnectivityGroupFlag(parser, api_type, required=False):
    """Adds connectivity flag group to the given parser."""
    if api_type == ApiType.CREATE:
        connectivity_group = parser.add_group("The connectivity method used by the migration job. If a connectivity method isn't specified, then it isn't added to the migration job.", mutex=True)
    elif api_type == ApiType.UPDATE:
        connectivity_group = parser.add_group("The connectivity method used by the migration job. If a connectivity method isn't specified, then it isn't updated for the migration job.", mutex=True)
    connectivity_group.add_argument('--static-ip', action='store_true', help='Use the default IP allowlist method. This method creates a public IP that will be used with the destination Cloud SQL database. The method works by configuring the source database server to accept connections from the outgoing IP of the Cloud SQL instance.')
    connectivity_group.add_argument('--peer-vpc', help='Name of the VPC network to peer with the Cloud SQL private network.')
    reverse_ssh_group = connectivity_group.add_group('Parameters for the reverse-SSH tunnel connectivity method.')
    reverse_ssh_group.add_argument('--vm-ip', help='Bastion Virtual Machine IP.', required=required)
    reverse_ssh_group.add_argument('--vm-port', help='Forwarding port for the SSH tunnel.', type=int, required=required)
    reverse_ssh_group.add_argument('--vm', help='Name of VM that will host the SSH tunnel bastion.')
    reverse_ssh_group.add_argument('--vpc', help='Name of the VPC network where the VM is hosted.', required=required)