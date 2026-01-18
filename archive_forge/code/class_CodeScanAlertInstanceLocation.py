from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class CodeScanAlertInstanceLocation(NonCompletableGithubObject):
    """
    This class represents code scanning alert instance locations.
    The reference can be found here https://docs.github.com/en/rest/reference/code-scanning.
    """

    def _initAttributes(self) -> None:
        self._path: Attribute[str] = NotSet
        self._start_line: Attribute[int] = NotSet
        self._start_column: Attribute[int] = NotSet
        self._end_line: Attribute[int] = NotSet
        self._end_column: Attribute[int] = NotSet

    def __str__(self) -> str:
        return f'{self.path} @ l{self.start_line}:c{self.start_column}-l{self.end_line}:c{self.end_column}'

    def __repr__(self) -> str:
        return self.get__repr__({'path': self.path, 'start_line': self.start_line, 'start_column': self.start_column, 'end_line': self.end_line, 'end_column': self.end_column})

    @property
    def path(self) -> str:
        return self._path.value

    @property
    def start_line(self) -> int:
        return self._start_line.value

    @property
    def start_column(self) -> int:
        return self._start_column.value

    @property
    def end_line(self) -> int:
        return self._end_line.value

    @property
    def end_column(self) -> int:
        return self._end_column.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'start_line' in attributes:
            self._start_line = self._makeIntAttribute(attributes['start_line'])
        if 'start_column' in attributes:
            self._start_column = self._makeIntAttribute(attributes['start_column'])
        if 'end_line' in attributes:
            self._end_line = self._makeIntAttribute(attributes['end_line'])
        if 'end_column' in attributes:
            self._end_column = self._makeIntAttribute(attributes['end_column'])