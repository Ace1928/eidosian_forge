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
def ApplyBackendSettings(self, backend_name):
    """Applies settings from the indicated backend to the `AppInfoExternal`.

    Backend entries can contain directives that modify other parts of the
    `app.yaml` file, such as the `start` directive, which adds a handler for the
    start request. This method performs those modifications.

    Args:
      backend_name: The name of a backend that is defined in the `backends`
          directive.

    Raises:
      BackendNotFound: If the indicated backend was not listed in the
          `backends` directive.
      DuplicateBackend: If the backend is found more than once in the `backends`
          directive.
    """
    if backend_name is None:
        return
    if self.backends is None:
        raise appinfo_errors.BackendNotFound
    self.version = backend_name
    match = None
    for backend in self.backends:
        if backend.name != backend_name:
            continue
        if match:
            raise appinfo_errors.DuplicateBackend
        else:
            match = backend
    if match is None:
        raise appinfo_errors.BackendNotFound
    if match.start is None:
        return
    start_handler = URLMap(url=_START_PATH, script=match.start)
    self.handlers.insert(0, start_handler)