from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class GitObject(NonCompletableGithubObject):
    """
    This class represents GitObjects
    """

    def _initAttributes(self) -> None:
        self._sha: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'sha': self._sha.value})

    @property
    def sha(self) -> str:
        return self._sha.value

    @property
    def type(self) -> str:
        return self._type.value

    @property
    def url(self) -> str:
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])