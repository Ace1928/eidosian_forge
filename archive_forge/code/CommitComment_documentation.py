from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList

        :calls: `DELETE /repos/{owner}/{repo}/comments/{comment_id}/reactions/{reaction_id}
                <https://docs.github.com/en/rest/reference/reactions#delete-a-commit-comment-reaction>`_
        :param reaction_id: integer
        :rtype: bool
        