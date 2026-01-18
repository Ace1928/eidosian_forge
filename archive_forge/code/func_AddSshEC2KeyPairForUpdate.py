from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSshEC2KeyPairForUpdate(parser, kind='control plane'):
    """Adds SSH config EC2 key pair related flags for update."""
    group = parser.add_group('SSH config', mutex=True)
    AddSshEC2KeyPair(group, kind)
    AddClearSshEc2KeyPair(group, kind)