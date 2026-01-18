from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
class _ObjectMapper(object):
    """Wrapper used for mapping attributes from a yaml file to an object.

  This wrapper is required because objects do not know what property they are
  associated with a creation time, and therefore can not be instantiated
  with the correct class until they are mapped to their parents.
  """

    def __init__(self):
        """Object mapper starts off with empty value."""
        self.value = None
        self.seen = set()

    def set_value(self, value):
        """Set value of instance to map to.

    Args:
      value: Instance that this mapper maps to.
    """
        self.value = value

    def see(self, key):
        if key in self.seen:
            raise yaml_errors.DuplicateAttribute("Duplicate attribute '%s'." % key)
        self.seen.add(key)