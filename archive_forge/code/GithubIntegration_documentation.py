from __future__ import annotations
import urllib.parse
import warnings
from typing import Any
import deprecated
import urllib3
from urllib3 import Retry
import github
from github import Consts
from github.Auth import AppAuth
from github.GithubApp import GithubApp
from github.GithubException import GithubException
from github.Installation import Installation
from github.InstallationAuthorization import InstallationAuthorization
from github.PaginatedList import PaginatedList
from github.Requester import Requester

        :calls: `GET /app <https://docs.github.com/en/rest/reference/apps#get-the-authenticated-app>`_
        