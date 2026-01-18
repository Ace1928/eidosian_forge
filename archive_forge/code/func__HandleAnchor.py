from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def _HandleAnchor(self, event):
    """Handle anchor attached to event.

    Currently will raise an error if anchor is used.  Anchors are used to
    define a document wide tag to a given value (scalar, mapping or sequence).

    Args:
      event: Event which may have anchor property set.

    Raises:
      NotImplementedError if event attempts to use an anchor.
    """
    if hasattr(event, 'anchor') and event.anchor is not None:
        raise NotImplementedError('Anchors not supported in this handler')