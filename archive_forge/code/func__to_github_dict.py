from __future__ import annotations
from typing import Any, Union
from typing_extensions import TypedDict
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
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