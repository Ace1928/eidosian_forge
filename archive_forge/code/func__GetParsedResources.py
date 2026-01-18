from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _GetParsedResources(self, fallthroughs_map, parsed_args):
    """Helper method to get the parsed resources using actively specified args.
    """
    types = []
    for concept_type in self.type_enum:
        try:
            concept_spec = self._name_to_concepts[concept_type.name]
            parsed_resource = concept_spec.Initialize(fallthroughs_map, parsed_args=parsed_args)
            types.append(TypedConceptResult(parsed_resource, concept_type))
        except concepts.InitializationError:
            continue
    return types