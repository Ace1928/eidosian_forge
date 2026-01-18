from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDiskSizeFlag(parser):
    return parser.add_argument('--disk-size', default='250GB', type=arg_parsers.BinarySize(lower_bound='20GB', upper_bound='2000GB', suggested_binary_size_scales=['GB']), help='      Configures the root volume size of your Compute Engine VM (in GB). The\n      minimum size is 20GB and the maximum is 2000GB. Specified value must be an\n      integer multiple of Gigabytes.\n      ')