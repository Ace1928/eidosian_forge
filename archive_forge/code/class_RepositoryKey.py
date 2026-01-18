from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class RepositoryKey(CompletableGithubObject):
    """
    This class represents RepositoryKeys. The reference can be found here https://docs.github.com/en/rest/reference/repos#deploy-keys
    """

    def _initAttributes(self) -> None:
        self._created_at: Attribute[datetime] = NotSet
        self._id: Attribute[int] = NotSet
        self._key: Attribute[str] = NotSet
        self._title: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet
        self._verified: Attribute[bool] = NotSet
        self._read_only: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'title': self._title.value})

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def key(self) -> str:
        self._completeIfNotSet(self._key)
        return self._key.value

    @property
    def title(self) -> str:
        self._completeIfNotSet(self._title)
        return self._title.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def verified(self) -> bool:
        self._completeIfNotSet(self._verified)
        return self._verified.value

    @property
    def read_only(self) -> bool:
        self._completeIfNotSet(self._read_only)
        return self._read_only.value

    def delete(self) -> None:
        """
        :calls: `DELETE /repos/{owner}/{repo}/keys/{id} <https://docs.github.com/en/rest/reference/repos#deploy-keys>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'key' in attributes:
            self._key = self._makeStringAttribute(attributes['key'])
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'verified' in attributes:
            self._verified = self._makeBoolAttribute(attributes['verified'])
        if 'read_only' in attributes:
            self._read_only = self._makeBoolAttribute(attributes['read_only'])