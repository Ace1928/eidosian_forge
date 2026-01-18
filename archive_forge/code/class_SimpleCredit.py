from __future__ import annotations
from typing import Any, Union
from typing_extensions import TypedDict
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class SimpleCredit(TypedDict):
    """
    A simple credit for a security advisory.
    """
    login: str | github.NamedUser.NamedUser
    type: str