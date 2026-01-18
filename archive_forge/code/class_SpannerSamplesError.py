from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
class SpannerSamplesError(exceptions.Error):
    """User error running Cloud Spanner sample app commands."""