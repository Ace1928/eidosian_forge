from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddModuleArgument(parser):
    parser.add_argument('--module', required=True, metavar='MODULE_NAME', help='Module name in Security Command Center')