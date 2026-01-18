from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

    This class represents repository invitations. The reference can be found here https://docs.github.com/en/rest/reference/repos#invitations
    