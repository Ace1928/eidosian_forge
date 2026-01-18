import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
class MatchDetail(object):
    """
	The :class:`.MatchDetail` class contains information about
	"""
    __slots__ = ('patterns',)

    def __init__(self, patterns):
        """
		Initialize the :class:`.MatchDetail` instance.

		*patterns* (:class:`~collections.abc.Sequence` of :class:`~pathspec.pattern.Pattern`)
		contains the patterns that matched the file in the order they were
		encountered.
		"""
        self.patterns = patterns
        '\n\t\t*patterns* (:class:`~collections.abc.Sequence` of :class:`~pathspec.pattern.Pattern`)\n\t\tcontains the patterns that matched the file in the order they were\n\t\tencountered.\n\t\t'