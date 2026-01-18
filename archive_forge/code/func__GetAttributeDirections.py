from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _GetAttributeDirections(attributes, full_fallthroughs_map):
    """Aggregates directions on how to set resource attribute."""
    directions = []
    for i, attribute in enumerate(attributes):
        fallthroughs = full_fallthroughs_map.get(attribute.name, [])
        tab = ' ' * 4
        to_specify = f'{i + 1}. To provide [{attribute.name}] attribute, do one of the following:'
        hints = (f'\n{tab}- {hint}' for hint in deps_lib.GetHints(fallthroughs))
        directions.append(to_specify + ''.join(hints))
    return '\n\n'.join(directions)