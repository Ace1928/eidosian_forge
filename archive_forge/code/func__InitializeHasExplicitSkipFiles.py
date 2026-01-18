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
def _InitializeHasExplicitSkipFiles(self, file_path, parsed):
    """Read app.yaml to determine whether user explicitly defined skip_files."""
    if getattr(parsed, 'skip_files', None) == appinfo.DEFAULT_SKIP_FILES:
        try:
            contents = files.ReadFileContents(file_path)
        except files.Error:
            contents = ''
        self._has_explicit_skip_files = 'skip_files' in contents
    else:
        self._has_explicit_skip_files = True