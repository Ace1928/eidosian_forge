from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def _FormDestinationUri(bucket):
    """Forms destination bucket uri."""
    return 'gs://{}/dependencies'.format(bucket)