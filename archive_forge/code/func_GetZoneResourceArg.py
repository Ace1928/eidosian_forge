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
def GetZoneResourceArg(help_text, positional=True, plural=False):
    arg_name = 'zones' if plural else 'zone'
    return concept_parsers.ConceptParser.ForResource(arg_name if positional else '--{}'.format(arg_name), GetZoneResourceSpec(), help_text, plural=plural, required=True)