from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Branch
import github.Commit
import github.GithubObject
import github.NamedUser
import github.Tag
import github.WorkflowRun
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList

        :calls: `GET /repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs <https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28#list-workflow-runs-for-a-workflow>`_
        