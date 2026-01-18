from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddRdmaArg(parser):
    """Adds the --rdma flag."""
    parser.add_argument('--rdma', hidden=True, action=arg_parsers.StoreTrueFalseAction, help='Enable/disable RDMA on this network.')