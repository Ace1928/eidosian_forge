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
def GetNormalizedLibraries(self):
    """Returns a list of normalized `Library` instances for this configuration.

    Returns:
      The list of active `Library` instances for this configuration. This
      includes directly-specified libraries, their required dependencies, and
      any libraries enabled by default. Any libraries with `latest` as their
      version will be replaced with the latest available version.
    """
    libraries = self.GetAllLibraries()
    enabled_libraries = set((library.name for library in libraries))
    for library in _SUPPORTED_LIBRARIES:
        if library.default_version and library.name not in enabled_libraries:
            libraries.append(Library(name=library.name, version=library.default_version))
    return libraries