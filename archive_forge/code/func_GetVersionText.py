from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.updater import installers
import requests
from six.moves import StringIO
def GetVersionText(self, version):
    """Gets the release notes text for the given version.

    Args:
      version: str, The version to get the release notes for.

    Returns:
      str, The release notes or None if the version does not exist.
    """
    index = self._GetVersionIndex(version)
    if index is None:
        return None
    return self._versions[index][1]