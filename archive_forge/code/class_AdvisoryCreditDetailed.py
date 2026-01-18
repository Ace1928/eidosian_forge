from __future__ import annotations
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class AdvisoryCreditDetailed(NonCompletableGithubObject):
    """
    This class represents a credit that is assigned to a SecurityAdvisory.
    The reference can be found here https://docs.github.com/en/rest/security-advisories/repository-advisories
    """

    @property
    def state(self) -> str:
        """
        :type: string
        """
        return self._state.value

    @property
    def type(self) -> str:
        """
        :type: string
        """
        return self._type.value

    @property
    def user(self) -> github.NamedUser.NamedUser:
        """
        :type: :class:`github.NamedUser.NamedUser`
        """
        return self._user.value

    def _initAttributes(self) -> None:
        self._state: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._user: Attribute[github.NamedUser.NamedUser] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'state' in attributes:
            self._state = self._makeStringAttribute(attributes['state'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'user' in attributes:
            self._user = self._makeClassAttribute(github.NamedUser.NamedUser, attributes['user'])