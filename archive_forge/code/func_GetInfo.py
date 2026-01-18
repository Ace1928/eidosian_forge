from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def GetInfo(self, presentation_spec_name):
    """Build ConceptInfo object for the spec with the given name."""
    if presentation_spec_name not in self.specs:
        raise ValueError('Presentation spec with name [{}] has not been added to the concept parser, cannot generate info.'.format(presentation_spec_name))
    presentation_spec = self.specs[presentation_spec_name]
    fallthroughs_map = {}
    for attribute in presentation_spec.concept_spec.attributes:
        fallthrough_strings = self._command_level_fallthroughs.get(presentation_spec.name, {}).get(attribute.name, [])
        fallthroughs = [self._MakeFallthrough(fallthrough_string) for fallthrough_string in fallthrough_strings]
        fallthroughs_map[attribute.name] = fallthroughs + attribute.fallthroughs
    return presentation_spec._GenerateInfo(fallthroughs_map)