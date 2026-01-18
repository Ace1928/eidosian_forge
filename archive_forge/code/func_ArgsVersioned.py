from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
@staticmethod
def ArgsVersioned(parser, version='v1'):
    perimeters.AddResourceArg(parser, 'to update')
    perimeters.AddUpdateDirectionalPoliciesGroupArgs(parser, version)
    repeated.AddPrimitiveArgs(parser, 'Service Perimeter', 'resources', 'Resources', include_set=False)
    repeated.AddPrimitiveArgs(parser, 'Service Perimeter', 'restricted-services', 'Restricted Services', include_set=False)
    repeated.AddPrimitiveArgs(parser, 'Service Perimeter', 'access-levels', 'Access Level', include_set=False)
    vpc_group = parser.add_argument_group('Arguments for configuring VPC accessible service restrictions.')
    vpc_group.add_argument('--enable-vpc-accessible-services', action='store_true', help="When specified restrict API calls within the Service Perimeter to the\n        set of vpc allowed services. To disable use\n        '--no-enable-vpc-accessible-services'.")
    repeated.AddPrimitiveArgs(vpc_group, 'Service Perimeter', 'vpc-allowed-services', 'VPC Allowed Services', include_set=False)
    parser.add_argument('--async', action='store_true', help='Return immediately, without waiting for the operation in\n                progress to complete.')