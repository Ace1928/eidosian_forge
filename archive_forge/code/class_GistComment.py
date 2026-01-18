from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class GistComment(CompletableGithubObject):
    """
    This class represents GistComments. The reference can be found here https://docs.github.com/en/rest/reference/gists#comments
    """

    def _initAttributes(self) -> None:
        self._body: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._id: Attribute[int] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'user': self._user.value})

    @property
    def body(self) -> str:
        self._completeIfNotSet(self._body)
        return self._body.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def updated_at(self) -> datetime:
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def user(self) -> github.NamedUser.NamedUser:
        self._completeIfNotSet(self._user)
        return self._user.value

    def delete(self) -> None:
        """
        :calls: `DELETE /gists/{gist_id}/comments/{id} <https://docs.github.com/en/rest/reference/gists#comments>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def edit(self, body: str) -> None:
        """
        :calls: `PATCH /gists/{gist_id}/comments/{id} <https://docs.github.com/en/rest/reference/gists#comments>`_
        """
        assert isinstance(body, str), body
        post_parameters = {'body': body}
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
        self._useAttributes(data)

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'body' in attributes:
            self._body = self._makeStringAttribute(attributes['body'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])