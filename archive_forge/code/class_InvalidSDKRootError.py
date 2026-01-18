from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
class InvalidSDKRootError(Error):
    """Error for when the root of the Cloud SDK is invalid or cannot be found."""

    def __init__(self):
        super(InvalidSDKRootError, self).__init__('The components management action could not be performed because the installation root of the Cloud SDK could not be located. If you previously used the Cloud SDK installer, you could re-install the SDK and retry again.')