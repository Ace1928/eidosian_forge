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
class NetrcAuth(HTTPBasicAuth, WithRequester['NetrcAuth']):
    """
    This class is used to authenticate via .netrc.
    """

    def __init__(self) -> None:
        super().__init__()
        self._login: Optional[str] = None
        self._password: Optional[str] = None

    @property
    def username(self) -> str:
        return self.login

    @property
    def login(self) -> str:
        assert self._login is not None, 'Method withRequester(Requester) must be called first'
        return self._login

    @property
    def password(self) -> str:
        assert self._password is not None, 'Method withRequester(Requester) must be called first'
        return self._password

    def withRequester(self, requester: Requester) -> 'NetrcAuth':
        super().withRequester(requester)
        auth = utils.get_netrc_auth(requester.base_url, raise_errors=True)
        if auth is None:
            raise RuntimeError(f'Could not get credentials from netrc for host {requester.hostname}')
        self._login, self._password = auth
        return self