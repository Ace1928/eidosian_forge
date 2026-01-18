from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class GitAuthor(NonCompletableGithubObject):
    """
    This class represents GitAuthors
    """

    def _initAttributes(self) -> None:
        self._name: Attribute[str] = NotSet
        self._email: Attribute[str] = NotSet
        self._date: Attribute[datetime] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'name': self._name.value})

    @property
    def date(self) -> datetime:
        return self._date.value

    @property
    def email(self) -> str:
        return self._email.value

    @property
    def name(self) -> str:
        return self._name.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'date' in attributes:
            self._date = self._makeDatetimeAttribute(attributes['date'])
        if 'email' in attributes:
            self._email = self._makeStringAttribute(attributes['email'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])