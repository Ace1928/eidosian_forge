from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
def _json_dump_recursion_helper(metadata):
    """See _get_json_dump docstring."""
    if isinstance(metadata, list):
        return [_json_dump_recursion_helper(item) for item in metadata]
    if not isinstance(metadata, dict):
        return resource_util.convert_to_json_parsable_type(metadata)
    formatted_dict = collections.OrderedDict(sorted(metadata.items()))
    for key, value in formatted_dict.items():
        if isinstance(value, dict):
            formatted_dict[key] = _json_dump_recursion_helper(value)
        elif isinstance(value, list):
            formatted_list = [_json_dump_recursion_helper(item) for item in value]
            if formatted_list:
                formatted_dict[key] = formatted_list
        elif value or resource_util.should_preserve_falsy_metadata_value(value):
            formatted_dict[key] = resource_util.convert_to_json_parsable_type(value)
    return formatted_dict