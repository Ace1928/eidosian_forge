from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _BuildFileReference(self, filename, use_basename=True):
    """Build a FileReference pointing to a file in GCS."""
    if not filename:
        return None
    if use_basename:
        filename = os.path.basename(filename)
    path = os.path.join(self._gcs_results_root, filename)
    return self._messages.FileReference(gcsPath=path)