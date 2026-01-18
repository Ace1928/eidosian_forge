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
def GetDnsSecStateFlagMapper(messages):
    return arg_utils.ChoiceEnumMapper('--dnssec-state', messages.ManagedZoneDnsSecConfig.StateValueValuesEnum, custom_mappings={'off': ('off', 'Disable DNSSEC for the managed zone.'), 'on': ('on', 'Enable DNSSEC for the managed zone.'), 'transfer': ('transfer', 'Enable DNSSEC and allow transferring a signed zone in or out.')}, help_str='The DNSSEC state for this managed zone.')