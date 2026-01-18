from __future__ import absolute_import
import copy
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
def HandleEvent(self, event, loader=None):
    """Handle individual PyYAML event.

    Args:
      event: Event to forward to method call in method call.

    Raises:
      IllegalEvent when receives an unrecognized or unsupported event type.
    """
    if event.__class__ not in _EVENT_METHOD_MAP:
        raise yaml_errors.IllegalEvent('%s is not a valid PyYAML class' % event.__class__.__name__)
    if event.__class__ in self._event_method_map:
        self._event_method_map[event.__class__](event, loader)