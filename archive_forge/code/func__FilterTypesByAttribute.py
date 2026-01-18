from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _FilterTypesByAttribute(self, attribute_info, concept_result):
    """Fitlers out types that do not contain actively specified attribute."""
    possible_types = []
    for candidate in concept_result:
        for attribute in attribute_info:
            if candidate.concept_type not in self._attribute_to_types_map.get(attribute.name, []):
                break
        else:
            possible_types.append(candidate)
    return possible_types