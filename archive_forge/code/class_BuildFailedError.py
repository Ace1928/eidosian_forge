from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
class BuildFailedError(exceptions.Error):
    """Raised when a Google Cloud Builder build fails."""