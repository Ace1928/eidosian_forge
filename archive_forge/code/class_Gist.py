from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GistComment
import github.GistFile
import github.GistHistoryState
import github.GithubObject
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, _NotSetType, is_defined, is_optional
from github.PaginatedList import PaginatedList
class Gist(CompletableGithubObject):
    """
    This class represents Gists. The reference can be found here https://docs.github.com/en/rest/reference/gists
    """

    def _initAttributes(self) -> None:
        self._comments: Attribute[int] = NotSet
        self._comments_url: Attribute[str] = NotSet
        self._commits_url: Attribute[str] = NotSet
        self._created_at: Attribute[datetime] = NotSet
        self._description: Attribute[str] = NotSet
        self._files: Attribute[dict[str, github.GistFile.GistFile]] = NotSet
        self._fork_of: Attribute[Gist] = NotSet
        self._forks: Attribute[list[Gist]] = NotSet
        self._forks_url: Attribute[str] = NotSet
        self._git_pull_url: Attribute[str] = NotSet
        self._git_push_url: Attribute[str] = NotSet
        self._history: Attribute[list[GistHistoryState]] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._id: Attribute[str] = NotSet
        self._owner: Attribute[github.NamedUser.NamedUser] = NotSet
        self._public: Attribute[bool] = NotSet
        self._updated_at: Attribute[datetime] = NotSet
        self._url: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def comments(self) -> int:
        self._completeIfNotSet(self._comments)
        return self._comments.value

    @property
    def comments_url(self) -> str:
        self._completeIfNotSet(self._comments_url)
        return self._comments_url.value

    @property
    def commits_url(self) -> str:
        self._completeIfNotSet(self._commits_url)
        return self._commits_url.value

    @property
    def created_at(self) -> datetime:
        self._completeIfNotSet(self._created_at)
        return self._created_at.value

    @property
    def description(self) -> str:
        self._completeIfNotSet(self._description)
        return self._description.value

    @property
    def files(self) -> dict[str, github.GistFile.GistFile]:
        self._completeIfNeeded()
        return self._files.value

    @property
    def fork_of(self) -> github.Gist.Gist:
        self._completeIfNotSet(self._fork_of)
        return self._fork_of.value

    @property
    def forks(self) -> list[Gist]:
        self._completeIfNotSet(self._forks)
        return self._forks.value

    @property
    def forks_url(self) -> str:
        self._completeIfNotSet(self._forks_url)
        return self._forks_url.value

    @property
    def git_pull_url(self) -> str:
        self._completeIfNotSet(self._git_pull_url)
        return self._git_pull_url.value

    @property
    def git_push_url(self) -> str:
        self._completeIfNotSet(self._git_push_url)
        return self._git_push_url.value

    @property
    def history(self) -> list[GistHistoryState]:
        self._completeIfNotSet(self._history)
        return self._history.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def id(self) -> str:
        self._completeIfNotSet(self._id)
        return self._id.value

    @property
    def owner(self) -> github.NamedUser.NamedUser:
        self._completeIfNotSet(self._owner)
        return self._owner.value

    @property
    def public(self) -> bool:
        self._completeIfNotSet(self._public)
        return self._public.value

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

    def create_comment(self, body: str) -> GistComment:
        """
        :calls: `POST /gists/{gist_id}/comments <https://docs.github.com/en/rest/reference/gists#comments>`_
        """
        assert isinstance(body, str), body
        post_parameters = {'body': body}
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/comments', input=post_parameters)
        return github.GistComment.GistComment(self._requester, headers, data, completed=True)

    def create_fork(self) -> Gist:
        """
        :calls: `POST /gists/{id}/forks <https://docs.github.com/en/rest/reference/gists>`_
        """
        headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/forks')
        return Gist(self._requester, headers, data, completed=True)

    def delete(self) -> None:
        """
        :calls: `DELETE /gists/{id} <https://docs.github.com/en/rest/reference/gists>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)

    def edit(self, description: Opt[str]=NotSet, files: Opt[dict[str, InputFileContent | None]]=NotSet) -> None:
        """
        :calls: `PATCH /gists/{id} <https://docs.github.com/en/rest/reference/gists>`_
        """
        assert is_optional(description, str), description
        assert isinstance(files, _NotSetType) or all((element is None or isinstance(element, github.InputFileContent) for element in files.values())), files
        post_parameters: dict[str, Any] = {}
        if is_defined(description):
            post_parameters['description'] = description
        if is_defined(files):
            post_parameters['files'] = {key: None if value is None else value._identity for key, value in files.items()}
        headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=post_parameters)
        self._useAttributes(data)

    def get_comment(self, id: int) -> GistComment:
        """
        :calls: `GET /gists/{gist_id}/comments/{id} <https://docs.github.com/en/rest/reference/gists#comments>`_
        """
        assert isinstance(id, int), id
        headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/comments/{id}')
        return github.GistComment.GistComment(self._requester, headers, data, completed=True)

    def get_comments(self) -> PaginatedList[GistComment]:
        """
        :calls: `GET /gists/{gist_id}/comments <https://docs.github.com/en/rest/reference/gists#comments>`_
        """
        return PaginatedList(github.GistComment.GistComment, self._requester, f'{self.url}/comments', None)

    def is_starred(self) -> bool:
        """
        :calls: `GET /gists/{id}/star <https://docs.github.com/en/rest/reference/gists>`_
        """
        status, headers, data = self._requester.requestJson('GET', f'{self.url}/star')
        return status == 204

    def reset_starred(self) -> None:
        """
        :calls: `DELETE /gists/{id}/star <https://docs.github.com/en/rest/reference/gists>`_
        """
        headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/star')

    def set_starred(self) -> None:
        """
        :calls: `PUT /gists/{id}/star <https://docs.github.com/en/rest/reference/gists>`_
        """
        headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/star')

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'comments' in attributes:
            self._comments = self._makeIntAttribute(attributes['comments'])
        if 'comments_url' in attributes:
            self._comments_url = self._makeStringAttribute(attributes['comments_url'])
        if 'commits_url' in attributes:
            self._commits_url = self._makeStringAttribute(attributes['commits_url'])
        if 'created_at' in attributes:
            self._created_at = self._makeDatetimeAttribute(attributes['created_at'])
        if 'description' in attributes:
            self._description = self._makeStringAttribute(attributes['description'])
        if 'files' in attributes:
            self._files = self._makeDictOfStringsToClassesAttribute(github.GistFile.GistFile, attributes['files'])
        if 'fork_of' in attributes:
            self._fork_of = self._makeClassAttribute(Gist, attributes['fork_of'])
        if 'forks' in attributes:
            self._forks = self._makeListOfClassesAttribute(Gist, attributes['forks'])
        if 'forks_url' in attributes:
            self._forks_url = self._makeStringAttribute(attributes['forks_url'])
        if 'git_pull_url' in attributes:
            self._git_pull_url = self._makeStringAttribute(attributes['git_pull_url'])
        if 'git_push_url' in attributes:
            self._git_push_url = self._makeStringAttribute(attributes['git_push_url'])
        if 'history' in attributes:
            self._history = self._makeListOfClassesAttribute(github.GistHistoryState.GistHistoryState, attributes['history'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'id' in attributes:
            self._id = self._makeStringAttribute(attributes['id'])
        if 'owner' in attributes:
            self._owner = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['owner'])
        if 'public' in attributes:
            self._public = self._makeBoolAttribute(attributes['public'])
        if 'updated_at' in attributes:
            self._updated_at = self._makeDatetimeAttribute(attributes['updated_at'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])