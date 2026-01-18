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
class FileChooser(object):
    """A FileChooser determines which files in a directory to upload.

  It's a fancy way of constructing a predicate (IsIncluded) along with a
  convenience method for walking a directory (GetIncludedFiles) and listing
  files to be uploaded based on that predicate.

  How the predicate operates is based on a gcloudignore file (see module
  docstring for details).
  """
    _INCLUDE_DIRECTIVE = '!include:'

    def __init__(self, patterns):
        self.patterns = patterns

    def IsIncluded(self, path, is_dir=False):
        """Returns whether the given file/directory should be included.

    This is determined according to the rules at
    https://git-scm.com/docs/gitignore except that symlinks are followed.

    In particular:
    - the method goes through pattern-by-pattern in-order
    - any matches of a parent directory on a particular pattern propagate to its
      children
    - if a parent directory is ignored, its children cannot be re-included

    Args:
      path: str, the path (relative to the root upload directory) to test.
      is_dir: bool, whether the path is a directory (or symlink to a directory).

    Returns:
      bool, whether the file should be uploaded
    """
        path_prefixes = glob.GetPathPrefixes(path)[1:]
        for path_prefix in path_prefixes:
            prefix_match = Match.NO_MATCH
            for pattern in self.patterns:
                is_prefix_dir = path_prefix != path or is_dir
                match = pattern.Matches(path_prefix, is_dir=is_prefix_dir)
                if match is not Match.NO_MATCH:
                    prefix_match = match
            if prefix_match is Match.IGNORE:
                log.debug('Skipping file [{}]'.format(path))
                return False
        return True

    def _RaiseOnSymlinkLoop(self, full_path):
        """Raise SymlinkLoopError if the given path is a symlink loop."""
        if not os.path.islink(encoding.Encode(full_path, encoding='utf-8')):
            return
        p = os.readlink(full_path)
        targets = set()
        while os.path.islink(p):
            if p in targets:
                raise SymlinkLoopError('The symlink [{}] refers to itself.'.format(full_path))
            targets.add(p)
            p = os.readlink(p)
        p = os.path.dirname(full_path)
        while p and os.path.basename(p):
            if os.path.samefile(p, full_path):
                raise SymlinkLoopError('The symlink [{}] refers to its own containing directory.'.format(full_path))
            p = os.path.dirname(p)

    def GetIncludedFiles(self, upload_directory, include_dirs=True):
        """Yields the files in the given directory that this FileChooser includes.

    Args:
      upload_directory: str, the path of the directory to upload.
      include_dirs: bool, whether to include directories

    Yields:
      str, the files and directories that should be uploaded.
    Raises:
      SymlinkLoopError: if there is a symlink referring to its own containing
      dir or itself.
    """
        for dirpath, orig_dirnames, filenames in os.walk(six.ensure_str(upload_directory), followlinks=True):
            dirpath = encoding.Decode(dirpath)
            dirnames = [encoding.Decode(dirname) for dirname in orig_dirnames]
            filenames = [encoding.Decode(filename) for filename in filenames]
            if dirpath == upload_directory:
                relpath = ''
            else:
                relpath = os.path.relpath(dirpath, upload_directory)
            for filename in filenames:
                file_relpath = os.path.join(relpath, filename)
                self._RaiseOnSymlinkLoop(os.path.join(dirpath, filename))
                if self.IsIncluded(file_relpath):
                    yield file_relpath
            for dirname in dirnames:
                file_relpath = os.path.join(relpath, dirname)
                full_path = os.path.join(dirpath, dirname)
                if self.IsIncluded(file_relpath, is_dir=True):
                    self._RaiseOnSymlinkLoop(full_path)
                    if include_dirs:
                        yield file_relpath
                else:
                    orig_dirnames.remove(dirname)

    @classmethod
    def FromString(cls, text, recurse=0, dirname=None):
        """Constructs a FileChooser from the given string.

    See `gcloud topic gcloudignore` for details.

    Args:
      text: str, the string (many lines, in the format specified in the
        documentation).
      recurse: int, how many layers of "#!include" directives to respect. 0
        means don't respect the directives, 1 means to respect the directives,
        but *not* in any "#!include"d files, etc.
      dirname: str, the base directory from which to "#!include"

    Raises:
      BadIncludedFileError: if a file being included does not exist or is not
        in the same directory.

    Returns:
      FileChooser.
    """
        patterns = []
        for line in text.splitlines():
            if line.startswith('#'):
                if line[1:].lstrip().startswith(cls._INCLUDE_DIRECTIVE):
                    patterns.extend(cls._GetIncludedPatterns(line, dirname, recurse))
                continue
            try:
                patterns.append(Pattern.FromString(line))
            except glob.InvalidLineError:
                pass
        return cls(patterns)

    @classmethod
    def _GetIncludedPatterns(cls, line, dirname, recurse):
        """Gets the patterns from an '#!include' line.

    Args:
      line: str, the line containing the '#!include' directive
      dirname: str, the name of the base directory from which to include files
      recurse: int, how many layers of "#!include" directives to respect. 0
        means don't respect the directives, 1 means to respect the directives,
        but *not* in any "#!include"d files, etc.

    Returns:
      list of Pattern, the patterns recursively included from the specified
        file.

    Raises:
      ValueError: if dirname is not provided
      BadIncludedFileError: if the file being included does not exist or is not
        in the same directory.
    """
        if not dirname:
            raise ValueError('dirname must be provided in order to include a file.')
        start_idx = line.find(cls._INCLUDE_DIRECTIVE)
        included_file = line[start_idx + len(cls._INCLUDE_DIRECTIVE):]
        if _GCLOUDIGNORE_PATH_SEP in included_file:
            raise BadIncludedFileError('May only include files in the same directory.')
        if not recurse:
            log.info('Not respecting `#!include` directive: [%s].', line)
            return []
        included_path = os.path.join(dirname, included_file)
        try:
            return cls.FromFile(included_path, recurse - 1).patterns
        except BadFileError as err:
            raise BadIncludedFileError(six.text_type(err))

    @classmethod
    def FromFile(cls, ignore_file_path, recurse=1):
        """Constructs a FileChooser from the given file path.

    See `gcloud topic gcloudignore` for details.

    Args:
      ignore_file_path: str, the path to the file in .gcloudignore format.
      recurse: int, how many layers of "#!include" directives to respect. 0
        means don't respect the directives, 1 means to respect the directives,
        but *not* in any "#!include"d files, etc.

    Raises:
      BadIncludedFileError: if the file being included does not exist or is not
        in the same directory.

    Returns:
      FileChooser.
    """
        try:
            text = files.ReadFileContents(ignore_file_path)
        except files.Error as err:
            raise BadFileError('Could not read ignore file [{}]: {}'.format(ignore_file_path, err))
        return cls.FromString(text, dirname=os.path.dirname(ignore_file_path), recurse=recurse)