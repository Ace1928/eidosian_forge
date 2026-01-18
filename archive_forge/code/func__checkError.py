from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
@staticmethod
def _checkError(headers: dict[str, Any], data: Any) -> tuple[dict[str, Any], Any]:
    if isinstance(data, dict) and 'error' in data:
        if data['error'] == 'bad_verification_code':
            raise BadCredentialsException(200, data, headers)
        raise GithubException(200, data, headers)
    return (headers, data)