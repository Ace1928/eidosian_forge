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
def AddToArchive(zip_file, src_dir, rel_path, is_file):
    """Add a file or directory (without its contents) to a ZIP archive.

  Args:
    zip_file: the ZIP archive
    src_dir: the base directory for rel_path, will not be recorded in the
      archive
    rel_path: the relative path to the file or directory to add
    is_file: a Boolean indicating whether rel_path points to a file (rather than
      a directory)
  """
    full_path = os.path.join(src_dir, rel_path)
    mtime = os.path.getmtime(full_path)
    if time.gmtime(mtime)[0] < 1980:
        if is_file:
            temp_file_handle, temp_file_path = tempfile.mkstemp()
            os.close(temp_file_handle)
            shutil.copyfile(full_path, temp_file_path)
            zip_file.write(temp_file_path, rel_path)
            os.remove(temp_file_path)
        else:
            with files.TemporaryDirectory() as temp_dir:
                zip_file.write(temp_dir, rel_path)
    else:
        zip_file.write(full_path, rel_path)