from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
import threading
import time
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import requests as creds_requests
from googlecloudsdk.core.util import encoding
import requests
def GetGCLLogTailer():
    """Return a GCL LogTailer."""
    try:
        from googlecloudsdk.api_lib.logging import tailing
    except ImportError:
        log.out.Print(LOG_STREAM_HELP_TEXT)
        return None
    return tailing.LogTailer()