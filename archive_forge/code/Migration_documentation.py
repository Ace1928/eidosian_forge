from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.PaginatedList
import github.Repository
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

        :calls: `DELETE /user/migrations/{migration_id}/repos/{repo_name}/lock <https://docs.github.com/en/rest/reference/migrations>`_
        