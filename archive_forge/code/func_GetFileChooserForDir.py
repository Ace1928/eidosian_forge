from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.util import glob
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import map  # pylint: disable=redefined-builtin
def GetFileChooserForDir(directory, default_ignore_file=DEFAULT_IGNORE_FILE, write_on_disk=True, gcloud_ignore_creation_predicate=_GitFilesExist, include_gitignore=True, ignore_file=None):
    """Gets the FileChooser object for the given directory.

  In order of preference:
  - If ignore_file is not none, use it to skip files.
    If the specified file does not exist, raise error.
  - Use .gcloudignore file in the top-level directory.
  - Evaluates creation predicate to determine whether to generate .gcloudignore.
    include_gitignore determines whether the generated .gcloudignore will
    include the user's .gitignore if one exists. If the directory is not
    writable, the file chooser corresponding to the ignore file that would have
    been generated is used.
  - If the creation predicate evaluates to false, returned FileChooser
    will choose all files.

  Args:
    directory: str, the path of the top-level directory to upload
    default_ignore_file: str, the ignore file to use if one is not found (and
      the directory has Git files).
    write_on_disk: bool, whether to save the generated gcloudignore to disk.
    gcloud_ignore_creation_predicate: one argument function, indicating if a
      .gcloudignore file should be created. The argument is the path of the
      directory that would contain the .gcloudignore file. By default
      .gcloudignore file will be created if and only if the directory contains
      .gitignore file or .git directory.
    include_gitignore: bool, whether the generated gcloudignore should include
      the user's .gitignore if present.
    ignore_file: custom ignore_file name.
              Override .gcloudignore file to customize files to be skipped.

  Raises:
    BadIncludedFileError: if a file being included does not exist or is not in
      the same directory.

  Returns:
    FileChooser: the FileChooser for the directory. If there is no .gcloudignore
    file and it can't be created the returned FileChooser will choose all files.
  """
    if ignore_file:
        gcloudignore_path = os.path.join(directory, ignore_file)
    else:
        if not properties.VALUES.gcloudignore.enabled.GetBool():
            log.info('Not using a .gcloudignore file since gcloudignore is globally disabled.')
            return FileChooser([])
        gcloudignore_path = os.path.join(directory, IGNORE_FILE_NAME)
    try:
        chooser = FileChooser.FromFile(gcloudignore_path)
    except BadFileError:
        pass
    else:
        log.info('Using ignore file at [{}].'.format(gcloudignore_path))
        return chooser
    if not gcloud_ignore_creation_predicate(directory):
        log.info('Not using ignore file.')
        return FileChooser([])
    ignore_contents = _GetIgnoreFileContents(default_ignore_file, directory, include_gitignore)
    log.info('Using default gcloudignore file:\n{0}\n{1}\n{0}'.format('--------------------------------------------------', ignore_contents))
    if write_on_disk:
        try:
            files.WriteFileContents(gcloudignore_path, ignore_contents, overwrite=False)
        except files.Error as err:
            log.info('Could not write .gcloudignore file: {}'.format(err))
        else:
            log.status.Print('Created .gcloudignore file. See `gcloud topic gcloudignore` for details.')
    return FileChooser.FromString(ignore_contents, recurse=1, dirname=directory)