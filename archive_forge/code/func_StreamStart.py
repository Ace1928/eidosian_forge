from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def StreamStart(self, event, loader):
    """Initializes internal state of handler

    Args:
      event: Ignored.
    """
    assert self._stack is None
    self._stack = []
    self._top = None
    self._results = []