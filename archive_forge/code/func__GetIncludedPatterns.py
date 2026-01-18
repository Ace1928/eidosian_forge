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