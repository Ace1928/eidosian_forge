from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _ConceptToName(self, concept_spec):
    """Helper to get the type enum name for a concept spec."""
    for name, spec in self._name_to_concepts.items():
        if spec == concept_spec:
            return name
    else:
        return None