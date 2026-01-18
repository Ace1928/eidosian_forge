from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
class ConflictingTypesError(Error):
    """Raised if there are multiple or no possible types for the spec."""

    def __init__(self, name, concept_specs, specified_attributes, fallthroughs_map):
        attributes = _GetAttrStr(specified_attributes)
        directions = _GetDirections(name, fallthroughs_map, concept_specs)
        message = f'Failed to determine type of [{name}] resource. You specified attributes [{attributes}].\n{directions}'
        super(ConflictingTypesError, self).__init__(message)