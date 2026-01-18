from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTpuNameOverrideArg(parser):
    return parser.add_argument('--name', help='      Override the name to use for VMs and TPUs (defaults to your username). ')