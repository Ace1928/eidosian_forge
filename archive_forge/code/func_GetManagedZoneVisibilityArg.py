from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetManagedZoneVisibilityArg():
    return base.Argument('--visibility', choices=['public', 'private'], default='public', help='Visibility of the zone. Public zones are visible to the public internet. Private zones are only visible in your internal networks denoted by the `--networks` flag.')