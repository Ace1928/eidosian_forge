from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class Stargazer(NonCompletableGithubObject):
    """
    This class represents Stargazers. The reference can be found here https://docs.github.com/en/rest/reference/activity#starring
    """

    def _initAttributes(self) -> None:
        self._starred_at: Attribute[datetime] = NotSet
        self._user: Attribute[NamedUser] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'user': self._user.value._login.value})

    @property
    def starred_at(self) -> datetime:
        return self._starred_at.value

    @property
    def user(self) -> NamedUser:
        return self._user.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'starred_at' in attributes:
            self._starred_at = self._makeDatetimeAttribute(attributes['starred_at'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])