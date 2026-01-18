from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddStoragePoolAllowAutoTieringArg(parser):
    """Adds the --allow-auto-tiering arg to the given parser."""
    parser.add_argument('--allow-auto-tiering', type=arg_parsers.ArgBoolean(truthy_strings=netapp_util.truthy, falsey_strings=netapp_util.falsey), help='Boolean flag indicating whether Storage Pool is allowed to use auto-tiering', hidden=True)