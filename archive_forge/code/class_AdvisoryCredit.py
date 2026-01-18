from __future__ import annotations
from typing import Any, Union
from typing_extensions import TypedDict
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class AdvisoryCredit(NonCompletableGithubObject):
    """
    This class represents a credit that is assigned to a SecurityAdvisory.
    The reference can be found here https://docs.github.com/en/rest/security-advisories/repository-advisories
    """

    @property
    def login(self) -> str:
        """
        :type: string
        """
        return self._login.value

    @property
    def type(self) -> str:
        """
        :type: string
        """
        return self._type.value

    def _initAttributes(self) -> None:
        self._login: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'login' in attributes:
            self._login = self._makeStringAttribute(attributes['login'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])

    @staticmethod
    def _validate_credit(credit: Credit) -> None:
        assert isinstance(credit, (dict, AdvisoryCredit)), credit
        if isinstance(credit, dict):
            assert 'login' in credit, credit
            assert 'type' in credit, credit
            assert isinstance(credit['login'], (str, github.NamedUser.NamedUser)), credit['login']
            assert isinstance(credit['type'], str), credit['type']
        else:
            assert isinstance(credit.login, str), credit.login
            assert isinstance(credit.type, str), credit.type

    @staticmethod
    def _to_github_dict(credit: Credit) -> SimpleCredit:
        assert isinstance(credit, (dict, AdvisoryCredit)), credit
        if isinstance(credit, dict):
            assert 'login' in credit, credit
            assert 'type' in credit, credit
            assert isinstance(credit['login'], (str, github.NamedUser.NamedUser)), credit['login']
            login = credit['login']
            if isinstance(login, github.NamedUser.NamedUser):
                login = login.login
            return {'login': login, 'type': credit['type']}
        else:
            return {'login': credit.login, 'type': credit.type}