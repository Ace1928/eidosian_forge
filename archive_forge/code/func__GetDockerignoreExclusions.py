from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import io
import operator
import os
import tarfile
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def _GetDockerignoreExclusions(upload_dir, gen_files):
    """Helper function to read the .dockerignore on disk or in generated files.

  Args:
    upload_dir: the path to the root directory.
    gen_files: dict of filename to contents of generated files.

  Returns:
    Set of exclusion expressions from the dockerignore file.
  """
    dockerignore = os.path.join(upload_dir, '.dockerignore')
    exclude = set()
    ignore_contents = None
    if os.path.exists(dockerignore):
        ignore_contents = files.ReadFileContents(dockerignore)
    else:
        ignore_contents = gen_files.get('.dockerignore')
    if ignore_contents:
        exclude = set(filter(bool, ignore_contents.splitlines()))
        exclude -= set(BLOCKLISTED_DOCKERIGNORE_PATHS)
    return exclude