import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def _GetFallthroughAndValue(attribute_name, fallthroughs_map, parsed_args):
    """Derives value and fallthrough used to derives value from map."""
    for possible_fallthrough in fallthroughs_map.get(attribute_name, []):
        try:
            value = possible_fallthrough.GetValue(parsed_args)
            return (value, possible_fallthrough)
        except deps_lib.FallthroughNotFoundError:
            continue
    else:
        return (None, None)