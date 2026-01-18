from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class RequiredStatusChecks(CompletableGithubObject):
    """
    This class represents Required Status Checks. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-status-checks-protection
    """

    def _initAttributes(self) -> None:
        self._strict: Attribute[bool] = NotSet
        self._contexts: Attribute[list[str]] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'strict': self._strict.value, 'url': self._url.value})

    @property
    def strict(self) -> bool:
        self._completeIfNotSet(self._strict)
        return self._strict.value

    @property
    def contexts(self) -> list[str]:
        self._completeIfNotSet(self._contexts)
        return self._contexts.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'strict' in attributes:
            self._strict = self._makeBoolAttribute(attributes['strict'])
        if 'contexts' in attributes:
            self._contexts = self._makeListOfStringsAttribute(attributes['contexts'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])