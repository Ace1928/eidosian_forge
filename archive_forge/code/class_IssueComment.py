from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
class IssueComment(CompletableGithubObject):
    """
    This class represents IssueComments. The reference can be found here https://docs.github.com/en/rest/reference/issues#comments
    """

    def _initAttributes(self) -> None:
        self._body: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._id: Attribute[int] = NotSet
        self._issue_url: Attribute[str] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet
        self._reactions: Attribute[dict] = NotSet

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
    def issue_url(self) -> str:
        self._completeIfNotSet(self._issue_url)
        return self._issue_url.value

    @property
    def updated_at(self) -> datetime:
        self._completeIfNotSet(self._updated_at)
        return self._updated_at.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def user(self) -> github.NamedUser.NamedUser:
        self._completeIfNotSet(self._user)
        return self._user.value

    @property
    def reactions(self) -> dict:
        self._completeIfNotSet(self._reactions)
        return self._reactions.value

    def delete(self) -> None:
        """
        :calls: `DELETE /repos/{owner}/{repo}/issues/comments/{id} <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def edit(self, body: str) -> None:
        """
        :calls: `PATCH /repos/{owner}/{repo}/issues/comments/{id} <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
        assert isinstance(body, str), body
        post_parameters = {'body': body}
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
        self._useAttributes(data)

    def get_reactions(self) -> PaginatedList[Reaction]:
        """
        :calls: `GET /repos/{owner}/{repo}/issues/comments/{id}/reactions
                <https://docs.github.com/en/rest/reference/reactions#list-reactions-for-an-issue-comment>`_
        """
        return PaginatedList(github.Reaction.Reaction, self._requester, f'{self.url}/reactions', None, headers={'Accept': Consts.mediaTypeReactionsPreview})

    def create_reaction(self, reaction_type: str) -> Reaction:
        """
        :calls: `POST /repos/{owner}/{repo}/issues/comments/{id}/reactions
                <https://docs.github.com/en/rest/reference/reactions#create-reaction-for-an-issue-comment>`_
        """
        assert isinstance(reaction_type, str), reaction_type
        post_parameters = {'content': reaction_type}
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/reactions', input=post_parameters, headers={'Accept': Consts.mediaTypeReactionsPreview})
        return github.Reaction.Reaction(self._requester, headers, data, completed=True)

    def delete_reaction(self, reaction_id: int) -> bool:
        """
        :calls: `DELETE /repos/{owner}/{repo}/issues/comments/{comment_id}/reactions/{reaction_id}
                <https://docs.github.com/en/rest/reference/reactions#delete-an-issue-comment-reaction>`_
        """
        assert isinstance(reaction_id, int), reaction_id
        status, _, _ = self._requester.requestJson('DELETE', f'{self.url}/reactions/{reaction_id}', headers={'Accept': Consts.mediaTypeReactionsPreview})
        return status == 204

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'body' in attributes:
            self._body = self._makeStringAttribute(attributes['body'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'issue_url' in attributes:
            self._issue_url = self._makeStringAttribute(attributes['issue_url'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])
        if 'reactions' in attributes:
            self._reactions = self._makeDictAttribute(attributes['reactions'])