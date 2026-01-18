import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
class JWT(Auth, ABC):
    """
    This class is the base class to authenticate with a JSON Web Token (JWT).
    https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app
    """

    @property
    def token_type(self) -> str:
        return 'Bearer'