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
def _CreateTar(upload_dir, gen_files, paths, gz):
    """Create tarfile for upload to GCS.

  The third-party code closes the tarfile after creating, which does not
  allow us to write generated files after calling docker.utils.tar
  since gzipped tarfiles can't be opened in append mode.

  Args:
    upload_dir: the directory to be archived
    gen_files: Generated files to write to the tar
    paths: allowed paths in the tarfile
    gz: gzipped tarfile object
  """
    root = os.path.abspath(upload_dir)
    t = tarfile.open(mode='w', fileobj=gz)
    for path in sorted(paths):
        full_path = os.path.join(root, path)
        t.add(full_path, arcname=path, recursive=False)
    for name, contents in six.iteritems(gen_files):
        genfileobj = io.BytesIO(contents.encode())
        tar_info = tarfile.TarInfo(name=name)
        tar_info.size = len(genfileobj.getvalue())
        t.addfile(tar_info, fileobj=genfileobj)
        genfileobj.close()
    t.close()