from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddUpgradeVersion(parser):
    parser.add_argument('--version', required=True, help='\n      Target cluster version to upgrade to. For example: "1.5.1".\n      ')