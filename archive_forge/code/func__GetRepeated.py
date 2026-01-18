from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def _GetRepeated(self, attribute):
    """Get the ultimate type of a repeated validator.

    Looks for an instance of validation.Repeated, returning its constructor.

    Args:
      attribute: Repeated validator attribute to find type for.

    Returns:
      The expected class of of the Type validator, otherwise object.
    """
    if isinstance(attribute, validation.Optional):
        attribute = attribute.validator
    if isinstance(attribute, validation.Repeated):
        return attribute.constructor
    return object