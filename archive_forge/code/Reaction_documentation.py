from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from . import Consts

        :calls: `DELETE /reactions/{id} <https://docs.github.com/en/rest/reference/reactions#delete-a-reaction-legacy>`_
        :rtype: None
        