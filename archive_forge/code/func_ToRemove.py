from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def ToRemove(self, update_seed):
    """Calculate the components that need to be uninstalled.

    Based on this given set of components, determine what we need to remove.
    When an update is done, we update all components connected to the initial
    set.  Based on this, we need to remove things that have been updated, or
    that no longer exist.  This method works with ToInstall().  For a given
    update set the update process should remove anything from ToRemove()
    followed by installing everything in ToInstall().  It is possible (and
    likely) that a component will be in both of these sets (when a new version
    is available).

    Args:
      update_seed: list of str, The component ids that we want to update.

    Returns:
      set of str, The component ids that should be removed.
    """
    if self._EnableFallback():
        connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
        connected |= self.latest.ConnectedComponents(connected | set(update_seed), platform_filter=self.__platform_filter)
        connected_darwin_x86_64 = self.current.ConnectedComponents(update_seed, platform_filter=self.DARWIN_X86_64)
        connected_darwin_x86_64 |= self.latest.ConnectedComponents(connected_darwin_x86_64 | set(update_seed), platform_filter=self.DARWIN_X86_64)
        connected |= connected_darwin_x86_64
        x86_removal_candidates = connected - self.FilterDuplicatesArm(connected)
        installed_components = set(self.current.components.keys())
        x86_removal_seed = x86_removal_candidates & installed_components
        if x86_removal_seed:
            log.warning('The ARM versions of the following components are available, replacing installed x86_64 versions: [{}].'.format(', '.join(x86_removal_seed)))
        removal_candidates = connected & set(self.current.components.keys())
        return (self.__removed_components | self.__updated_components | x86_removal_seed) & removal_candidates
    else:
        connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
        connected |= self.latest.ConnectedComponents(connected | set(update_seed), platform_filter=self.__platform_filter)
        removal_candidates = connected & set(self.current.components.keys())
        return (self.__removed_components | self.__updated_components) & removal_candidates