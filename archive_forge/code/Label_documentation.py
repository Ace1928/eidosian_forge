import urllib.parse
from typing import Any, Dict
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional

        :calls: `PATCH /repos/{owner}/{repo}/labels/{name} <https://docs.github.com/en/rest/reference/issues#labels>`_
        