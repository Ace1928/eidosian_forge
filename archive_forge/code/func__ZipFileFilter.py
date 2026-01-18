from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
def _ZipFileFilter(self, file_name):
    """Filter all files in the archive directory to only allow Apigee files."""
    if not file_name.startswith(self._ARCHIVE_ROOT):
        return False
    _, ext = os.path.splitext(file_name)
    full_path = os.path.join(self._src_dir, file_name)
    if os.path.basename(full_path).startswith('.'):
        return False
    if os.path.isdir(full_path):
        return True
    if os.path.isfile(full_path) and ext.lower() in self._APIGEE_ARCHIVE_FILE_EXTENSIONS:
        return True
    return False