from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.config import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
def _GetPropertiesToDisplay(self, args):
    """List available regular properties."""
    section, prop = properties.ParsePropertyString(args.property)
    if prop:
        return {section: {prop: properties.VALUES.Section(section).Property(prop).Get()}}
    if section:
        return {section: properties.VALUES.Section(section).AllValues(list_unset=args.all)}
    return properties.VALUES.AllValues(list_unset=args.all)