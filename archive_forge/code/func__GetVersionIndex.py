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
def _GetVersionIndex(self, version):
    """Gets the index of the given version in the list of parsed versions.

    Args:
      version: str, The version to get the index for.

    Returns:
      int, The index of the given version or None if not found.
    """
    for i, (v, _) in enumerate(self._versions):
        if v == version:
            return i
    return None