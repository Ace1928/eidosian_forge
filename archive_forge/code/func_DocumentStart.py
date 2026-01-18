from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def DocumentStart(self, event, loader):
    """Build new document.

    Pushes new document on to stack.

    Args:
      event: Ignored.
    """
    assert self._stack == []
    self._Push(_TOKEN_DOCUMENT, self._builder.BuildDocument())