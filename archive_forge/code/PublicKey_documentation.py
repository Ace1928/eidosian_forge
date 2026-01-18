from __future__ import annotations
from base64 import b64encode
from typing import Any
from nacl import encoding, public
from github.GithubObject import Attribute, CompletableGithubObject, NotSet

    This class represents either an organization public key or a repository public key.
    The reference can be found here https://docs.github.com/en/rest/reference/actions#get-an-organization-public-key
    or here https://docs.github.com/en/rest/reference/actions#get-a-repository-public-key
    