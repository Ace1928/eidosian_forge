from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddVolumeIops(parser, prefix):
    parser.add_argument('--{}-volume-iops'.format(prefix), type=int, help='Number of I/O operations per second (IOPS) to provision for the {} volume.'.format(prefix))