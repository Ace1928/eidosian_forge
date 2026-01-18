from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester

        :calls: `POST /login/oauth/access_token <https://docs.github.com/en/developers/apps/identifying-and-authorizing-users-for-github-apps>`_
        :param refresh_token: string
        