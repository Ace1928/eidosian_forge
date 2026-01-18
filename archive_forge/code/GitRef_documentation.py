from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.GitObject
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional

        :calls: `PATCH /repos/{owner}/{repo}/git/refs/{ref} <https://docs.github.com/en/rest/reference/git#references>`_
        