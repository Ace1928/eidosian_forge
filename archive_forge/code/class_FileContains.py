import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
class FileContains(Matcher):
    """Matches if the given file has the specified contents."""

    def __init__(self, contents=None, matcher=None):
        """Construct a ``FileContains`` matcher.

        Can be used in a basic mode where the file contents are compared for
        equality against the expected file contents (by passing ``contents``).
        Can also be used in a more advanced way where the file contents are
        matched against an arbitrary matcher (by passing ``matcher`` instead).

        :param contents: If specified, match the contents of the file with
            these contents.
        :param matcher: If specified, match the contents of the file against
            this matcher.
        """
        if contents == matcher is None:
            raise AssertionError('Must provide one of `contents` or `matcher`.')
        if None not in (contents, matcher):
            raise AssertionError('Must provide either `contents` or `matcher`, not both.')
        if matcher is None:
            self.matcher = Equals(contents)
        else:
            self.matcher = matcher

    def match(self, path):
        mismatch = PathExists().match(path)
        if mismatch is not None:
            return mismatch
        f = open(path)
        try:
            actual_contents = f.read()
            return self.matcher.match(actual_contents)
        finally:
            f.close()

    def __str__(self):
        return 'File at path exists and contains %s' % self.contents