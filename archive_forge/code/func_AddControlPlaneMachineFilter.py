from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddControlPlaneMachineFilter(parser):
    parser.add_argument('--control-plane-machine-filter', help='\n      Only machines matching this filter will be allowed to host\n      local control plane nodes.\n      The filtering language accepts strings like "name=<name>",\n      and is documented here: [AIP-160](https://google.aip.dev/160).\n      ')