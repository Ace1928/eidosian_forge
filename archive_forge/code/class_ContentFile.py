from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, _ValuedAttribute
class ContentFile(CompletableGithubObject):
    """
    This class represents ContentFiles. The reference can be found here https://docs.github.com/en/rest/reference/repos#contents
    """

    def _initAttributes(self) -> None:
        self._content: Attribute[str] = NotSet
        self._download_url: Attribute[str] = NotSet
        self._encoding: Attribute[str] = NotSet
        self._git_url: Attribute[str] = NotSet
        self._html_url: Attribute[str] = NotSet
        self._license: Attribute[License] = NotSet
        self._name: Attribute[str] = NotSet
        self._path: Attribute[str] = NotSet
        self._repository: Attribute[Repository] = NotSet
        self._sha: Attribute[str] = NotSet
        self._size: Attribute[int] = NotSet
        self._type: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet
        self._text_matches: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'path': self._path.value})

    @property
    def content(self) -> str:
        self._completeIfNotSet(self._content)
        return self._content.value

    @property
    def decoded_content(self) -> bytes:
        assert self.encoding == 'base64', f'unsupported encoding: {self.encoding}'
        return base64.b64decode(bytearray(self.content, 'utf-8'))

    @property
    def download_url(self) -> str:
        self._completeIfNotSet(self._download_url)
        return self._download_url.value

    @property
    def encoding(self) -> str:
        self._completeIfNotSet(self._encoding)
        return self._encoding.value

    @property
    def git_url(self) -> str:
        self._completeIfNotSet(self._git_url)
        return self._git_url.value

    @property
    def html_url(self) -> str:
        self._completeIfNotSet(self._html_url)
        return self._html_url.value

    @property
    def license(self) -> License:
        self._completeIfNotSet(self._license)
        return self._license.value

    @property
    def name(self) -> str:
        self._completeIfNotSet(self._name)
        return self._name.value

    @property
    def path(self) -> str:
        self._completeIfNotSet(self._path)
        return self._path.value

    @property
    def repository(self) -> Repository:
        if self._repository is NotSet:
            repo_url = '/'.join(self.url.split('/')[:6])
            self._repository = _ValuedAttribute(github.Repository.Repository(self._requester, self._headers, {'url': repo_url}, completed=False))
        return self._repository.value

    @property
    def sha(self) -> str:
        self._completeIfNotSet(self._sha)
        return self._sha.value

    @property
    def size(self) -> int:
        self._completeIfNotSet(self._size)
        return self._size.value

    @property
    def type(self) -> str:
        self._completeIfNotSet(self._type)
        return self._type.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def text_matches(self) -> str:
        self._completeIfNotSet(self._text_matches)
        return self._text_matches.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'content' in attributes:
            self._content = self._makeStringAttribute(attributes['content'])
        if 'download_url' in attributes:
            self._download_url = self._makeStringAttribute(attributes['download_url'])
        if 'encoding' in attributes:
            self._encoding = self._makeStringAttribute(attributes['encoding'])
        if 'git_url' in attributes:
            self._git_url = self._makeStringAttribute(attributes['git_url'])
        if 'html_url' in attributes:
            self._html_url = self._makeStringAttribute(attributes['html_url'])
        if 'license' in attributes:
            self._license = self._makeClassAttribute(github.License.License, attributes['license'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'repository' in attributes:
            self._repository = self._makeClassAttribute(github.Repository.Repository, attributes['repository'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'size' in attributes:
            self._size = self._makeIntAttribute(attributes['size'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])
        if 'text_matches' in attributes:
            self._text_matches = self._makeListOfDictsAttribute(attributes['text_matches'])