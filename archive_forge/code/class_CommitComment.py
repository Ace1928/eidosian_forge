from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
class CommitComment(CompletableGithubObject):
    """
    This class represents CommitComments. The reference can be found here https://docs.github.com/en/rest/reference/repos#comments
    """

    def _initAttributes(self) -> None:
        self._body: Attribute[str] = NotSet
        self._commit_id: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._id: Attribute[int] = NotSet
        self._line: Attribute[int] = NotSet
        self._path: Attribute[str] = NotSet
        self._position: Attribute[int] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value, 'user': self.user})

    @property
    def body(self) -> str:
        self._completeIfNotSet(self._body)
        return self._body.value

    @property
    def commit_id(self) -> str:
        self._completeIfNotSet(self._commit_id)
        return self._commit_id.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def id(self) -> int:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def line(self) -> int:
        self._completeIfNotSet(self._line)
        return self._line.value

    @property
    def path(self) -> str:
        self._completeIfNotSet(self._path)
        return self._path.value

    @property
    def position(self) -> int:
        self._completeIfNotSet(self._position)
        return self._position.value

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
        :calls: `DELETE /repos/{owner}/{repo}/comments/{id} <https://docs.github.com/en/rest/reference/repos#comments>`_
        :rtype: None
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def edit(self, body: str) -> None:
        """
        :calls: `PATCH /repos/{owner}/{repo}/comments/{id} <https://docs.github.com/en/rest/reference/repos#comments>`_
        """
        assert isinstance(body, str), body
        post_parameters = {'body': body}
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
        self._useAttributes(data)

    def get_reactions(self) -> PaginatedList[Reaction]:
        """
        :calls: `GET /repos/{owner}/{repo}/comments/{id}/reactions
                <https://docs.github.com/en/rest/reference/reactions#list-reactions-for-a-commit-comment>`_
        :return: :class: :class:`github.PaginatedList.PaginatedList` of :class:`github.Reaction.Reaction`
        """
        return PaginatedList(github.Reaction.Reaction, self._requester, f'{self.url}/reactions', None, headers={'Accept': Consts.mediaTypeReactionsPreview})

    def create_reaction(self, reaction_type: str) -> Reaction:
        """
        :calls: `POST /repos/{owner}/{repo}/comments/{id}/reactions
                <https://docs.github.com/en/rest/reference/reactions#create-reaction-for-a-commit-comment>`_
        """
        assert isinstance(reaction_type, str), reaction_type
        post_parameters = {'content': reaction_type}
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/reactions', input=post_parameters, headers={'Accept': Consts.mediaTypeReactionsPreview})
        return github.Reaction.Reaction(self._requester, headers, data, completed=True)

    def delete_reaction(self, reaction_id: int) -> bool:
        """
        :calls: `DELETE /repos/{owner}/{repo}/comments/{comment_id}/reactions/{reaction_id}
                <https://docs.github.com/en/rest/reference/reactions#delete-a-commit-comment-reaction>`_
        :param reaction_id: integer
        :rtype: bool
        """
        assert isinstance(reaction_id, int), reaction_id
        status, _, _ = self._requester.requestJson('DELETE', f'{self.url}/reactions/{reaction_id}', headers={'Accept': Consts.mediaTypeReactionsPreview})
        return status == 204

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'body' in attributes:
            self._body = self._makeStringAttribute(attributes['body'])
        if 'commit_id' in attributes:
            self._commit_id = self._makeStringAttribute(attributes['commit_id'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'line' in attributes:
            self._line = self._makeIntAttribute(attributes['line'])
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'position' in attributes:
            self._position = self._makeIntAttribute(attributes['position'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])