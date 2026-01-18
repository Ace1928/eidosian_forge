from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class StatsContributor(NonCompletableGithubObject):
    """
    This class represents StatsContributors. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-all-contributor-commit-activity
    """

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

    @property
    def author(self) -> github.NamedUser.NamedUser:
        return self._author.value

    @property
    def total(self) -> int:
        return self._total.value

    @property
    def weeks(self) -> list[Week]:
        return self._weeks.value

    def _initAttributes(self) -> None:
        self._author: Attribute[github.NamedUser.NamedUser] = NotSet
        self._total: Attribute[int] = NotSet
        self._weeks: Attribute[list[StatsContributor.Week]] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'author' in attributes:
            self._author = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['author'])
        if 'total' in attributes:
            self._total = self._makeIntAttribute(attributes['total'])
        if 'weeks' in attributes:
            self._weeks = self._makeListOfClassesAttribute(self.Week, attributes['weeks'])