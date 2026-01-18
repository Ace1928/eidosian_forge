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
def ToInstall(self, update_seed):
    """Calculate the components that need to be installed.

    Based on this given set of components, determine what we need to install.
    When an update is done, we update all components connected to the initial
    set.  Based on this, we need to install things that have been updated or
    that are new.  This method works with ToRemove().  For a given update set
    the update process should remove anything from ToRemove() followed by
    installing everything in ToInstall().  It is possible (and likely) that a
    component will be in both of these sets (when a new version is available).

    Args:
      update_seed: list of str, The component ids that we want to update.

    Returns:
      set of str, The component ids that should be removed.
    """
    installed_components = list(self.current.components.keys())
    missing_platform = self.latest.CheckMissingPlatformExecutable(update_seed, self.__platform_filter)
    if self._EnableFallback():
        missing_platform_darwin_x86_64 = self.latest.CheckMissingPlatformExecutable(update_seed, self.DARWIN_X86_64)
        native_valid_seed = self.__native_all_components - missing_platform
        native_seed = set(update_seed) & native_valid_seed
        darwin_x86_64 = set(update_seed) - native_seed
        darwin_x86_64 -= missing_platform_darwin_x86_64
        valid_seed = native_seed | darwin_x86_64
        platform_seeds = [c_id for c_id in darwin_x86_64 if 'darwin' not in c_id]
        if platform_seeds:
            log.warning('The ARM versions of the following components are not available yet, using x86_64 versions instead: [{}].'.format(', '.join(platform_seeds)))
        local_connected = self.current.ConnectedComponents(valid_seed, platform_filter=self.__platform_filter)
        all_required = self.latest.DependencyClosureForComponents(local_connected | set(valid_seed), platform_filter=self.__platform_filter)
        local_connected_darwin_x86_64 = self.current.ConnectedComponents(valid_seed, platform_filter=self.DARWIN_X86_64)
        all_required |= self.latest.DependencyClosureForComponents(local_connected_darwin_x86_64 | valid_seed, platform_filter=self.DARWIN_X86_64)
        remote_connected = self.latest.ConnectedComponents(local_connected | valid_seed, platform_filter=self.__platform_filter)
        remote_connected |= self.latest.ConnectedComponents(local_connected_darwin_x86_64 | valid_seed, platform_filter=self.__platform_filter)
        all_required |= remote_connected & set(installed_components)
        all_required = self.FilterDuplicatesArm(all_required)
        dep_missing_platform = self.latest.CheckMissingPlatformExecutable(all_required, self.DARWIN_X86_64)
        if dep_missing_platform:
            log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(dep_missing_platform)))
            all_required -= dep_missing_platform
    else:
        local_connected = self.current.ConnectedComponents(update_seed, platform_filter=self.__platform_filter)
        all_required = self.latest.DependencyClosureForComponents(local_connected | set(update_seed), platform_filter=self.__platform_filter)
        remote_connected = self.latest.ConnectedComponents(local_connected | set(update_seed), platform_filter=self.__platform_filter)
        all_required |= remote_connected & set(installed_components)
        dep_missing_platform = self.latest.CheckMissingPlatformExecutable(all_required, self.__platform_filter)
        if dep_missing_platform:
            log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(dep_missing_platform)))
            all_required -= dep_missing_platform
    different = self.__new_components | self.__updated_components
    return set((c for c in all_required if c in different or c not in installed_components))