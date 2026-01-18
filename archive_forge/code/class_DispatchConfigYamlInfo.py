from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
class DispatchConfigYamlInfo(ConfigYamlInfo):
    """Provides methods for getting 1p-ready representation."""

    def _HandlerToDict(self, handler):
        """Converst a dispatchinfo handler into a 1p-ready dict."""
        parsed_url = dispatchinfo.ParsedURL(handler.url)
        dispatch_domain = parsed_url.host
        if not parsed_url.host_exact:
            dispatch_domain = '*' + dispatch_domain
        dispatch_path = parsed_url.path
        if not parsed_url.path_exact:
            trailing_matcher = '/*' if dispatch_path.endswith('/') else '*'
            dispatch_path = dispatch_path.rstrip('/') + trailing_matcher
        return {'domain': dispatch_domain, 'path': dispatch_path, 'service': handler.service}

    def GetRules(self):
        """Get dispatch rules on a format suitable for Admin API.

    Returns:
      [{'service': str, 'domain': str, 'path': str}], rules.
    """
        return [self._HandlerToDict(h) for h in self.parsed.dispatch or []]