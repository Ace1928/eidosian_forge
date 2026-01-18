from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def CreateTopicResourcePresentationSpec(verb, help_text, group):
    """Create add_topic, remove_topic or update_topic specs."""
    name = '--' + verb + '-topic'
    return presentation_specs.ResourcePresentationSpec(name, GetTopicResourceSpec(), help_text, prefixes=True, group=group)