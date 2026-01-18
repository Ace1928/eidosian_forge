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
def _GetIncludedPaths(upload_dir, source_files, exclude):
    """Helper function to filter paths in root using dockerignore and skip_files.

  We iterate separately to filter on skip_files in order to preserve expected
  behavior (standard deployment skips directories if they contain only files
  ignored by skip_files).

  Args:
    upload_dir: the path to the root directory.
    source_files: [str], relative paths to upload.
    exclude: the .dockerignore file exclusions.

  Returns:
    Set of paths (relative to upload_dir) to include.
  """
    import docker
    root = os.path.abspath(upload_dir)
    paths = docker.utils.exclude_paths(root, list(exclude))
    paths.intersection_update(source_files)
    return paths