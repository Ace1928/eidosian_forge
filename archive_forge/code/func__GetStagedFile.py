from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
import six.moves.urllib.parse
def _GetStagedFile(self, file_str):
    """Validate file URI and register it for uploading if it is local."""
    drive, _ = os.path.splitdrive(file_str)
    uri = six.moves.urllib.parse.urlsplit(file_str, allow_fragments=False)
    is_local = drive or not uri.scheme
    if not is_local:
        return file_str
    if not os.path.exists(file_str):
        raise files.Error('File Not Found: [{0}].'.format(file_str))
    if self._staging_dir is None:
        raise exceptions.ArgumentError('Could not determine where to stage local file {0}. When submitting a job to a cluster selected via --cluster-labels, either\n- a staging bucket must be provided via the --bucket argument, or\n- all provided files must be non-local.'.format(file_str))
    basename = os.path.basename(file_str)
    self.files_to_stage.append(file_str)
    staged_file = six.moves.urllib.parse.urljoin(self._staging_dir, basename)
    return staged_file