from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def _ParseBucketName(name):
    """Normalizes bucket name.

  Normalizes bucket name. If it starts with gs://, remove it.
  Api_lib's function doesn't like the gs prefix.

  Args:
    name: gs bucket name string.

  Returns:
    A name string without 'gs://' prefix.
  """
    gs = 'gs://'
    if name.startswith(gs):
        return name[len(gs):]
    return name