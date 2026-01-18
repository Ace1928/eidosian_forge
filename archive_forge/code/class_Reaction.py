from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from . import Consts
class Reaction(CompletableGithubObject):
    """
    This class represents Reactions. The reference can be found here https://docs.github.com/en/rest/reference/reactions
    """

    def _initAttributes(self) -> None:
        self._content: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._id: Attribute[int] = NotSet
        self._user: Attribute[NamedUser] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'user': self._user.value})

    @property
    def content(self) -> str:
        self._completeIfNotSet(self._content)
        return self._content.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def user(self) -> NamedUser:
        self._completeIfNotSet(self._user)
        return self._user.value

    def delete(self) -> None:
        """
        :calls: `DELETE /reactions/{id} <https://docs.github.com/en/rest/reference/reactions#delete-a-reaction-legacy>`_
        :rtype: None
        """
        self._requester.requestJsonAndCheck('DELETE', f'{self._parentUrl('')}/reactions/{self.id}', headers={'Accept': Consts.mediaTypeReactionsPreview})

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'content' in attributes:
            self._content = self._makeStringAttribute(attributes['content'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])