from __future__ import annotations
from datetime import date, datetime
from typing import Any
import github.GithubObject
import github.Label
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList

        :calls: `GET /repos/{owner}/{repo}/milestones/{number}/labels <https://docs.github.com/en/rest/reference/issues#labels>`_
        