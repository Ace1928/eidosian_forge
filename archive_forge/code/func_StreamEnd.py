from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def StreamEnd(self, event, loader):
    """Cleans up internal state of handler after parsing

    Args:
      event: Ignored.
    """
    assert self._stack == [] and self._top is None
    self._stack = None