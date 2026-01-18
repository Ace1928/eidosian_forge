from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import info_holders
def _GenerateInfo(self, fallthroughs_map):
    """Gets the MultitypeResourceInfo object for the ConceptParser.

    Args:
      fallthroughs_map: {str: [googlecloudsdk.calliope.concepts.deps.
        _FallthroughBase]}, dict keyed by attribute name to lists of
        fallthroughs.

    Returns:
      info_holders.MultitypeResourceInfo, the ResourceInfo object.
    """
    return info_holders.MultitypeResourceInfo(self.name, self.concept_spec, self.group_help, self.attribute_to_args_map, fallthroughs_map, required=self.required, plural=self.plural, group=self.group)