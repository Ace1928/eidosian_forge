from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
class SpannerMigrationException(c_except.Error):
    """Base Exception for any errors raised by gcloud spanner migration surface."""