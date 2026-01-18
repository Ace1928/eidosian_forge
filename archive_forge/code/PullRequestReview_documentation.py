from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

        :calls: `PUT /repos/{owner}/{repo}/pulls/{number}/reviews/{review_id}
                <https://docs.github.com/en/rest/pulls/reviews#update-a-review-for-a-pull-request>`_
        