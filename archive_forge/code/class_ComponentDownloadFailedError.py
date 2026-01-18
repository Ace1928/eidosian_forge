from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import tarfile
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import local_file_adapter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import retry
import requests
import six
class ComponentDownloadFailedError(Error):
    """Exception for when we cannot download a component for some reason."""

    def __init__(self, component_id, e):
        super(ComponentDownloadFailedError, self).__init__('The component [{component_id}] failed to download.\n\n'.format(component_id=component_id) + six.text_type(e))