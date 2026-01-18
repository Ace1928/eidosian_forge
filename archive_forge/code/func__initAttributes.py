from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.GitObject
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
def _initAttributes(self) -> None:
    self._object: Attribute[GitObject] = NotSet
    self._ref: Attribute[str] = NotSet
    self._url: Attribute[str] = NotSet