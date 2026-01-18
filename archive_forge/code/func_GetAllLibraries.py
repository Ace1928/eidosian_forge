from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetAllLibraries(self):
    """Returns a list of all `Library` instances active for this configuration.

    Returns:
      The list of active `Library` instances for this configuration. This
      includes directly-specified libraries as well as any required
      dependencies.
    """
    if not self.libraries:
        return []
    library_names = set((library.name for library in self.libraries))
    required_libraries = []
    for library in self.libraries:
        for required_name, required_version in REQUIRED_LIBRARIES.get((library.name, library.version), []):
            if required_name not in library_names:
                required_libraries.append(Library(name=required_name, version=required_version))
    return [Library(**library.ToDict()) for library in self.libraries + required_libraries]