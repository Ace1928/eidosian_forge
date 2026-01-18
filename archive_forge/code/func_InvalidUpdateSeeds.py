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
def InvalidUpdateSeeds(self, component_ids):
    """Sees if any of the given components don't exist locally or remotely.

    Args:
      component_ids: list of str, The components that the user wants to update.

    Returns:
      set of str, The component ids that do not exist anywhere.
    """
    invalid_seeds = set(component_ids) - self.__all_components
    missing_platform = self.latest.CheckMissingPlatformExecutable(component_ids, self.__platform_filter)
    if self._EnableFallback():
        missing_platform_x86_64 = self.latest.CheckMissingPlatformExecutable(component_ids, self.DARWIN_X86_64)
        missing_platform &= missing_platform_x86_64
        native_invalid_ids = set(component_ids) - self.__native_all_components
        arm_x86_ids = native_invalid_ids & self.__darwin_x86_64_components
        if arm_x86_ids:
            rosetta2_installed = self._CheckRosetta2Exists()
            if not rosetta2_installed:
                log.warning('The ARM versions of the components [{}] are not available yet. To download and execute the x86_64 version of the components, please install Rosetta 2 first by running the command: softwareupdate --install-rosetta.'.format(', '.join(arm_x86_ids)))
                invalid_seeds |= arm_x86_ids
    if missing_platform:
        log.warning('The platform specific binary does not exist for components [{}].'.format(', '.join(missing_platform)))
    return invalid_seeds | missing_platform