from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRun
import github.GitCommit
import github.GithubApp
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_defined, is_optional
from github.PaginatedList import PaginatedList

        :calls: `GET /repos/{owner}/{repo}/check-suites/{check_suite_id}/check-runs <https://docs.github.com/en/rest/reference/checks#list-check-runs-in-a-check-suite>`_
        