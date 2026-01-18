from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AttributeConfig(name, fallthroughs=None, help_text=None, validate=False):
    """Returns a ResourceParameterAttributeConfig for the attribute named `name`.

  Args:
    name: singular name of the attribute. Must exist in ENTITIES.
    fallthroughs: optional list of gcloud fallthrough objects which should be
      used to get this attribute's value if the user doesn't specify one.
    help_text: help text to use for this resource parameter instead of the
      default help text for the attribute.
    validate: whether to check that user-provided value for this attribute
      matches the expected pattern.
  """
    validator = None
    if validate:
        validator = arg_parsers.RegexpValidator(_ValidPatternForEntity(name), 'Must match the format of a valid {2} ({3})'.format(*ENTITIES[name]))
    return concepts.ResourceParameterAttributeConfig(name=name, parameter_name=ENTITIES[name].plural, value_type=validator, help_text=help_text or ENTITIES[name].secondary_description, fallthroughs=fallthroughs)