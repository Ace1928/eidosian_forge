from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def BuildSequence(self, top_value):
    """Build a new sequence representation.

    Called when StartSequence event received.  Type of object is determined
    by Builder sub-class.

    Args:
      top_value: Object which will be new sequences parant.  Will be object
        returned from previous call to BuildMapping or BuildSequence.

    Returns:
      Instance of new object that represents a sequence type in target model.
    """