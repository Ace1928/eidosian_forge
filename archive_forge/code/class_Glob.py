from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
class Glob(object):
    """A file-matching glob pattern.

  See https://git-scm.com/docs/gitignore for full syntax specification.

  Attributes:
    pattern: str, a globbing pattern.
    must_be_dir: bool, true if only dirs match.
  """

    def __init__(self, pattern, must_be_dir=False):
        self.pattern = pattern
        self.must_be_dir = must_be_dir

    def _MatchesHelper(self, pattern_parts, path):
        """Determines whether the given pattern matches the given path.

    Args:
      pattern_parts: list of str, the list of pattern parts that must all match
        the path.
      path: str, the path to match.

    Returns:
      bool, whether the patch matches the pattern_parts (Matches() will convert
        this into a Match value).
    """
        if not pattern_parts:
            return True
        if path is None:
            return False
        pattern_part = pattern_parts[-1]
        remaining_pattern = pattern_parts[:-1]
        if path:
            path = os.path.normpath(path)
        remaining_path, path_part = os.path.split(path)
        if not path_part:
            remaining_path = None
        if pattern_part == '**':
            path_prefixes = GetPathPrefixes(path)
            if not (remaining_pattern and remaining_pattern[0] == ''):
                remaining_pattern.insert(0, '')
            return any((self._MatchesHelper(remaining_pattern, prefix) for prefix in path_prefixes))
        if pattern_part == '*' and (not remaining_pattern):
            if remaining_path and len(remaining_path) > 1:
                return False
        if not fnmatch.fnmatch(path_part, pattern_part):
            return False
        return self._MatchesHelper(remaining_pattern, remaining_path)

    def Matches(self, path, is_dir=False):
        """Returns a Match for this pattern and the given path."""
        if self.must_be_dir and (not is_dir):
            return False
        if self._MatchesHelper(self.pattern.split('/'), path):
            return True
        else:
            return False

    @classmethod
    def FromString(cls, line):
        """Creates a pattern for an individual line of an ignore file.

    Windows-style newlines must be removed.

    Args:
      line: str, The line to parse.

    Returns:
      Pattern.

    Raises:
      InvalidLineError: if the line was invalid (comment, blank, contains
        invalid consecutive stars).
    """
        if line.endswith('/'):
            line = line[:-1]
            must_be_dir = True
        else:
            must_be_dir = False
        line = _HandleSpaces(line)
        if re.search(_ENDS_IN_ODD_NUMBER_SLASHES_RE, line):
            raise InvalidLineError('Line [{}] ends in an odd number of [\\]s.'.format(line))
        line = _Unescape(line)
        if not line:
            raise InvalidLineError('Line [{}] is blank.'.format(line))
        return cls(line, must_be_dir=must_be_dir)