from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def EndMapping(self, top_value, mapping):
    """Previously constructed mapping scope is at an end.

    Called when the end of a mapping block is encountered.  Useful for
    additional clean up or end of scope validation.

    Args:
      top_value: Value which is parent of the mapping.
      mapping: Mapping which is at the end of its scope.
    """