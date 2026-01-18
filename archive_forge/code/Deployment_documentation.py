from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList

        :calls: `POST /repos/{owner}/{repo}/deployments/{deployment_id}/statuses <https://docs.github.com/en/rest/reference/repos#create-a-deployment-status>`_
        