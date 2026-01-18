from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class Week(NonCompletableGithubObject):
    """
        This class represents weekly statistics of a contributor.
        """

    @property
    def w(self) -> datetime:
        return self._w.value

    @property
    def a(self) -> int:
        return self._a.value

    @property
    def d(self) -> int:
        return self._d.value

    @property
    def c(self) -> int:
        return self._c.value

    def _initAttributes(self) -> None:
        self._w: Attribute[datetime] = NotSet
        self._a: Attribute[int] = NotSet
        self._d: Attribute[int] = NotSet
        self._c: Attribute[int] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'w' in attributes:
            self._w = self._makeTimestampAttribute(attributes['w'])
        if 'a' in attributes:
            self._a = self._makeIntAttribute(attributes['a'])
        if 'd' in attributes:
            self._d = self._makeIntAttribute(attributes['d'])
        if 'c' in attributes:
            self._c = self._makeIntAttribute(attributes['c'])