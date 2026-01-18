import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def AddFlagFallthroughs(base_fallthroughs_map, attributes, attribute_to_args_map):
    """Adds flag fallthroughs to fallthrough map.

  Iterates through each attribute and prepends a flag fallthrough.
  This allows resource attributes to be resolved to flag first. For example:

    {'book': [deps.ValueFallthrough('foo')]}

  will update to something like...

    {
        'book': [
            deps.ArgFallthrough('--foo'),
            deps.ValueFallthrough('foo')
        ]
    }

  Args:
    base_fallthroughs_map: {str: [deps._FallthroughBase]}, A map of attribute
      names to fallthroughs
    attributes: list[concepts.Attribute], list of attributes associated
      with the resource
    attribute_to_args_map: {str: str}, A map of attribute names to the names
      of their associated flags.
  """
    for attribute in attributes:
        current_fallthroughs = base_fallthroughs_map.get(attribute.name, [])
        if (arg_name := attribute_to_args_map.get(attribute.name)):
            arg_fallthrough = deps_lib.ArgFallthrough(arg_name)
        else:
            arg_fallthrough = None
        if arg_fallthrough:
            filtered_fallthroughs = [f for f in current_fallthroughs if f != arg_fallthrough]
            fallthroughs = [arg_fallthrough] + filtered_fallthroughs
        else:
            fallthroughs = current_fallthroughs
        base_fallthroughs_map[attribute.name] = fallthroughs