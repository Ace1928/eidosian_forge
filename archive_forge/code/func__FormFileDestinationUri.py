from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def _FormFileDestinationUri(destination, uri):
    """Forms uri representing uploaded file."""
    return os.path.join(destination, os.path.basename(uri))