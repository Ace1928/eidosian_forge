from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import hashlib
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import times
from googlecloudsdk.third_party.appengine.tools import context_util
from six.moves import map  # pylint: disable=redefined-builtin
Reformat a filename for the deployment manifest if it is Windows format.