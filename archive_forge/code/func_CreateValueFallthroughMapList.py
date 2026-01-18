import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def CreateValueFallthroughMapList(base_fallthroughs_map, attribute_name, parsed_args):
    """Generates a list of fallthrough maps for each anchor value in a list.

  For each anchor value, generate a fallthrough map. For example, if user
  provides anchor values ['foo', 'bar'] and a base fallthrough like...

    {'book': [deps.ArgFallthrough('--book')]}

  will generate somehting like...

    [
        {'book': [deps.ValueFallthrough('foo')]},
        {'book': [deps.ValueFallthrough('bar')]}
    ]

  Args:
    base_fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs we are updating
    attribute_name: str, entry in fallthrough map we are updating
    parsed_args: Namespace | None, used to derive the value for ValueFallthrough

  Returns:
    list[{str: deps._FallthroughBase}], a list of fallthrough maps for
    each parsed anchor value
  """
    attribute_values, attribute_fallthrough = _GetFallthroughAndValue(attribute_name, base_fallthroughs_map, parsed_args)
    map_list = []
    if not attribute_fallthrough:
        return map_list
    for value in attribute_values:
        new_map = {**base_fallthroughs_map}
        _UpdateMapWithValueFallthrough(new_map, value, attribute_name, attribute_fallthrough)
        map_list.append(new_map)
    return map_list