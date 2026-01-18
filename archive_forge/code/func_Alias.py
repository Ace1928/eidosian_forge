from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def Alias(self, event, loader):
    """Not implemented yet.

    Args:
      event: Ignored.
    """
    raise NotImplementedError('References not supported in this handler')