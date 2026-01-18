from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def SequenceEnd(self, event, loader):
    """End of sequence.

    Args:
      event: Ignored
      loader: Ignored.
      """
    assert self._top[0] == _TOKEN_SEQUENCE
    end_object = self._Pop()
    top_value = self._top[1]
    self._builder.EndSequence(top_value, end_object)