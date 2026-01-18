from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTpuNameArg(parser):
    return parser.add_argument('execution_group_name', help='      The execution group name to delete. ')