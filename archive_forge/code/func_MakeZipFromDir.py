from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import tempfile
import time
import zipfile
import googlecloudsdk.core.util.files as files
import six
def MakeZipFromDir(dest_zip_file, src_dir, predicate=None):
    """Create a ZIP archive from a directory.

  This is similar to shutil.make_archive. However, prior to Python 3.8,
  shutil.make_archive cannot create ZIP archives for files with mtimes older
  than 1980. So that's why this function exists.

  Examples:
    Filesystem:
    /tmp/a/
    /tmp/b/B

    >>> MakeZipFromDir('my.zip', '/tmp')
    Creates zip with content:
    a/
    b/B

  Note this is caller responsibility to use appropriate platform-dependent
  path separator.

  Note filenames containing path separator are supported.

  Args:
    dest_zip_file: str, filesystem path to the zip file to be created. Note that
      directory should already exist for destination zip file.
    src_dir: str, filesystem path to the directory to zip up
    predicate: callable, takes one argument (file path). File will be included
               in the zip if and only if the predicate(file_path). Defaults to
               always true.
  """
    if predicate is None:
        predicate = lambda x: True
    zip_file = zipfile.ZipFile(dest_zip_file, 'w', _ZIP_COMPRESSION)
    try:
        for root, _, filelist in os.walk(six.text_type(src_dir)):
            dir_path = os.path.normpath(os.path.relpath(root, src_dir))
            if not predicate(dir_path):
                continue
            if dir_path != os.curdir:
                AddToArchive(zip_file, src_dir, dir_path, False)
            for file_name in filelist:
                file_path = os.path.join(dir_path, file_name)
                if not predicate(file_path):
                    continue
                AddToArchive(zip_file, src_dir, file_path, True)
    finally:
        zip_file.close()