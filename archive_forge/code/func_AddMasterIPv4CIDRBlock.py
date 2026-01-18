from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddMasterIPv4CIDRBlock(parser):
    """Adds --master-ipv4-cidr-block flag."""
    parser.add_argument('--master-ipv4-cidr-block', help="The /28 network that the control plane will use. Defaults to '172.16.0.128/28' if flag is not provided.")