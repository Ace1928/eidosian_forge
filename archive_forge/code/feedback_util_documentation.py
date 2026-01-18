from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
Opens a new tab in the web browser to the new issue page for Cloud SDK.

  The page will be pre-populated with relevant information.

  Args:
    info: InfoHolder, the data from of `gcloud info`
    log_data: LogData, parsed representation of a recent log
  