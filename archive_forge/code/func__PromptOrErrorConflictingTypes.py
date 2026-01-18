from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def _PromptOrErrorConflictingTypes(self, specified_attributes, full_fallthroughs_map, parsed_resources):
    """If one or more type is parsed, send prompt for user to confirm.

    If user is unable to confirm resource type, raise ConflictingTypesError

    Args:
      specified_attributes: list[Attribute], list of explicitly specified
        resource attributes
      full_fallthroughs_map: {str: [deps_lib._FallthroughBase]}, a dict of
        finalized fallthroughs for the resource.
      parsed_resources: list[TypedConceptResult], list of parsed resources

    Returns:
      concepts.Resource, resource user elects to specify

    Raises:
      ConflictingTypesError: if user is not able to specify preferred resource.
    """
    if not console_io.CanPrompt():
        raise ConflictingTypesError(self.name, self._concept_specs, specified_attributes, full_fallthroughs_map)
    guess_list = [guess.result.RelativeName() for guess in parsed_resources]
    attr_str = _GetAttrStr(specified_attributes)
    try:
        selected_index = console_io.PromptChoice(guess_list, message=f'Failed determine type of [{self.name}] resource. You specified attributes [{attr_str}].\nDid you mean to specify one of the following resources?', prompt_string='Please enter your numeric choice. Defaults to', cancel_option=True, default=len(guess_list))
    except console_io.OperationCancelledError:
        raise ConflictingTypesError(self.name, self._concept_specs, specified_attributes, full_fallthroughs_map)
    return parsed_resources[selected_index]