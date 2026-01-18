from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetInstancePlacement(args):
    instance_placement = getattr(args, 'instance_placement', None)
    return _TenancyEnumMapper().GetEnumForChoice(instance_placement) if instance_placement else None