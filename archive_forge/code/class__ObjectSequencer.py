from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_builder
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
class _ObjectSequencer(object):
    """Wrapper used for building sequences from a yaml file to a list.

  This wrapper is required because objects do not know what property they are
  associated with a creation time, and therefore can not be instantiated
  with the correct class until they are mapped to their parents.
  """

    def __init__(self):
        """Object sequencer starts off with empty value."""
        self.value = []
        self.constructor = None

    def set_constructor(self, constructor):
        """Set object used for constructing new sequence instances.

    Args:
      constructor: Callable which can accept no arguments.  Must return
        an instance of the appropriate class for the container.
    """
        self.constructor = constructor