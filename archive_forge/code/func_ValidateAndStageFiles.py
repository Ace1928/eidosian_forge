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
def ValidateAndStageFiles(self):
    """Validate file URIs and upload them if they are local."""
    for file_type, file_or_files in six.iteritems(self.files_by_type):
        if not file_or_files:
            continue
        elif isinstance(file_or_files, six.string_types):
            self.files_by_type[file_type] = self._GetStagedFile(file_or_files)
        else:
            staged_files = [self._GetStagedFile(f) for f in file_or_files]
            self.files_by_type[file_type] = staged_files
    if self.files_to_stage:
        log.info('Staging local files {0} to {1}.'.format(self.files_to_stage, self._staging_dir))
        storage_helpers.Upload(self.files_to_stage, self._staging_dir)