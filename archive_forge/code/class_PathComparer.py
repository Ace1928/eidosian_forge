import difflib
import pathlib
from typing import Any, List, Union
class PathComparer:
    """
    OS-independent path comparison.

    Windows path sep and posix path sep:

    >>> '\\to\\index' == PathComparer('/to/index')
    True
    >>> '\\to\\index' == PathComparer('/to/index2')
    False

    Windows path with drive letters

    >>> 'C:\\to\\index' == PathComparer('/to/index')
    True
    >>> 'C:\\to\\index' == PathComparer('C:/to/index')
    True
    >>> 'C:\\to\\index' == PathComparer('D:/to/index')
    False
    """

    def __init__(self, path: Union[str, pathlib.Path]):
        """
        :param str path: path string, it will be cast as pathlib.Path.
        """
        self.path = pathlib.Path(path)

    def __str__(self) -> str:
        return self.path.as_posix()

    def __repr__(self) -> str:
        return "<{0.__class__.__name__}: '{0}'>".format(self)

    def __eq__(self, other: Union[str, pathlib.Path]) -> bool:
        return not bool(self.ldiff(other))

    def diff(self, other: Union[str, pathlib.Path]) -> List[str]:
        """compare self and other.

        When different is not exist, return empty list.

        >>> PathComparer('/to/index').diff('C:\\to\\index')
        []

        When different is exist, return unified diff style list as:

        >>> PathComparer('/to/index').diff('C:\\to\\index2')
        [
           '- C:/to/index'
           '+ C:/to/index2'
           '?            +'
        ]
        """
        return self.ldiff(other)

    def ldiff(self, other: Union[str, pathlib.Path]) -> List[str]:
        return self._diff(self.path, pathlib.Path(other))

    def rdiff(self, other: Union[str, pathlib.Path]) -> List[str]:
        return self._diff(pathlib.Path(other), self.path)

    def _diff(self, lhs: pathlib.Path, rhs: pathlib.Path) -> List[str]:
        if lhs == rhs:
            return []
        if lhs.drive or rhs.drive:
            s_path, o_path = (lhs.absolute().as_posix(), rhs.absolute().as_posix())
        else:
            s_path, o_path = (lhs.as_posix(), rhs.as_posix())
        if s_path == o_path:
            return []
        return [line.strip() for line in difflib.Differ().compare([s_path], [o_path])]