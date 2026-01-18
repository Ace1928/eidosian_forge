from typing import (
from .pathspec import (
from .pattern import (
from .patterns.gitwildmatch import (
from .util import (
@staticmethod
def _match_file(patterns: Iterable[Tuple[int, GitWildMatchPattern]], file: str) -> Tuple[Optional[bool], Optional[int]]:
    """
		Check the file against the patterns.

		.. NOTE:: Subclasses of :class:`~pathspec.pathspec.PathSpec` may override
		   this method as an instance method. It does not have to be a static
		   method. The signature for this method is subject to change.

		*patterns* (:class:`~collections.abc.Iterable`) yields each indexed pattern
		(:class:`tuple`) which contains the pattern index (:class:`int`) and actual
		pattern (:class:`~pathspec.pattern.Pattern`).

		*file* (:class:`str`) is the normalized file path to be matched against
		*patterns*.

		Returns a :class:`tuple` containing whether to include *file* (:class:`bool`
		or :data:`None`), and the index of the last matched pattern (:class:`int` or
		:data:`None`).
		"""
    out_include: Optional[bool] = None
    out_index: Optional[int] = None
    out_priority = 0
    for index, pattern in patterns:
        if pattern.include is not None:
            match = pattern.match_file(file)
            if match is not None:
                dir_mark = match.match.groupdict().get(_DIR_MARK)
                if dir_mark:
                    priority = 1
                else:
                    priority = 2
                if pattern.include and dir_mark:
                    out_include = pattern.include
                    out_index = index
                    out_priority = priority
                elif priority >= out_priority:
                    out_include = pattern.include
                    out_index = index
                    out_priority = priority
    return (out_include, out_index)