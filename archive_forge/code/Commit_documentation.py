from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CheckRun
import github.CheckSuite
import github.CommitCombinedStatus
import github.CommitComment
import github.CommitStats
import github.CommitStatus
import github.File
import github.GitCommit
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList

        :class: `GET /repos/{owner}/{repo}/commits/{ref}/check-suites <https://docs.github.com/en/rest/reference/checks#list-check-suites-for-a-git-reference>`_
        