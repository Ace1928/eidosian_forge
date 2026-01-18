from __future__ import annotations
from datetime import datetime
from os.path import basename
from typing import Any, BinaryIO
import github.GitReleaseAsset
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts

        :calls: `GET /repos/{owner}/{repo}/releases/{release_id}/assets <https://docs.github.com/en/rest/reference/repos#list-release-assets>`_
        