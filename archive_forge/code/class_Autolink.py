from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class Autolink(NonCompletableGithubObject):
    """
    This class represents Repository autolinks.
    The reference can be found here https://docs.github.com/en/rest/repos/autolinks?apiVersion=2022-11-28
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[int] = NotSet
        self._key_prefix: Attribute[str] = NotSet
        self._url_template: Attribute[str] = NotSet
        self._is_alphanumeric: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def id(self) -> int:
        return self._id.value

    @property
    def key_prefix(self) -> str:
        return self._key_prefix.value

    @property
    def url_template(self) -> str:
        return self._url_template.value

    @property
    def is_alphanumeric(self) -> bool:
        return self._is_alphanumeric.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'key_prefix' in attributes:
            self._key_prefix = self._makeStringAttribute(attributes['key_prefix'])
        if 'url_template' in attributes:
            self._url_template = self._makeStringAttribute(attributes['url_template'])
        if 'is_alphanumeric' in attributes:
            self._is_alphanumeric = self._makeBoolAttribute(attributes['is_alphanumeric'])